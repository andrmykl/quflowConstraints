"""Import SVG curves and centre markers as QuFlow matrices.

The public entry points are re-exported by :mod:`quflow.constraints`.  By
default, SVG geometry is mapped to QuFlow's complete Hammer plot; a 2:1 page
gives matching visual proportions, and a compact stereographic compatibility
mode is also available.  Every closed curve becomes the regular zero level of
its own mountain-shaped skew-Hermitian QuFlow matrix, while native Inkscape
spirals can be used as centre annotations for QuFlow blob matrices.
"""

from __future__ import annotations

import math
import os

import numpy as np
from matplotlib.path import Path as MatplotlibPath
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from ._svg import _read_svg_curves, _read_svg_spiral_centers


__all__ = ["import_svg_blobs", "import_svg_domain"]


def _positive_integer(value, name):
    """Return *value* as a positive integer."""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _finite_scalar(value, name):
    """Return *value* as a finite real float."""
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    return result


def _center_coordinates(center):
    """Return a validated latitude/longitude pair in degrees."""
    try:
        latitude, longitude = center
    except (TypeError, ValueError) as error:
        raise ValueError(
            "center must be a (latitude_degrees, longitude_degrees) pair."
        ) from error
    latitude = _finite_scalar(latitude, "center latitude")
    longitude = _finite_scalar(longitude, "center longitude")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError("center latitude must lie between -90 and 90 degrees.")
    return latitude, longitude


def _chart_basis(center):
    """Return east, north, centre as the columns of a rotation matrix."""
    latitude, longitude = _center_coordinates(center)

    latitude, longitude = np.deg2rad((latitude, longitude))
    east = np.array((-np.sin(longitude), np.cos(longitude), 0.0))
    north = np.array(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    centre = np.array(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        )
    )
    return np.column_stack((east, north, centre))


def _mapping_options(mapping, center, chart_radius_degrees):
    """Validate shared SVG-to-sphere mapping options."""
    if not isinstance(mapping, str) or mapping not in {
        "hammer",
        "stereographic",
    }:
        raise ValueError(
            "mapping must be either 'hammer' or 'stereographic'."
        )

    chart_radius_degrees = _finite_scalar(
        chart_radius_degrees,
        "chart_radius_degrees",
    )
    if not 0.0 < chart_radius_degrees < 90.0:
        raise ValueError(
            "chart_radius_degrees must lie strictly between 0 and 90."
        )
    center_coordinates = _center_coordinates(center)
    if mapping == "hammer" and (
        center_coordinates != (0.0, 0.0)
        or chart_radius_degrees != 60.0
    ):
        raise ValueError(
            "center and chart_radius_degrees are only available with "
            "mapping='stereographic'."
        )
    return center_coordinates, chart_radius_degrees


def _stereographic_sampling_coordinates(
    construction_N,
    raster_size,
    chart_scale,
    basis,
):
    """Return image coordinates for the stereographic QuFlow-grid samples."""
    import quflow as qf

    theta, phi = qf.sphgrid(construction_N)
    points = np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )
    local = points @ basis
    denominator = 1.0 + local[..., 2]
    u = np.zeros_like(denominator)
    v = np.zeros_like(denominator)
    valid_denominator = denominator > 32.0 * np.finfo(float).eps
    np.divide(
        local[..., 0],
        denominator,
        out=u,
        where=valid_denominator,
    )
    np.divide(
        local[..., 1],
        denominator,
        out=v,
        where=valid_denominator,
    )

    column = (1.0 + u / chart_scale) * (raster_size - 1) / 2.0
    row = (1.0 - v / chart_scale) * (raster_size - 1) / 2.0
    valid = (
        valid_denominator
        & (row >= 0.0)
        & (row <= raster_size - 1)
        & (column >= 0.0)
        & (column <= raster_size - 1)
    )
    return points, row, column, valid


def _hammer_coordinates(longitude, latitude):
    """Project displayed longitude/latitude to normalized Hammer coordinates."""
    cosine_latitude = np.cos(latitude)
    denominator = np.sqrt(
        1.0 + cosine_latitude * np.cos(longitude / 2.0)
    )
    horizontal = (
        cosine_latitude * np.sin(longitude / 2.0) / denominator
    )
    vertical = np.sin(latitude) / denominator
    return horizontal, vertical


def _inverse_hammer_coordinates(horizontal, vertical):
    """Invert normalized Hammer coordinates inside its unit ellipse."""
    radius_squared = horizontal**2 + vertical**2
    tolerance = 64.0 * np.finfo(float).eps
    if np.any(radius_squared > 1.0 + tolerance):
        raise ValueError("SVG points must lie inside the Hammer ellipse.")

    radius_squared = np.minimum(radius_squared, 1.0)
    z = np.sqrt(1.0 - radius_squared / 2.0)
    longitude = 2.0 * np.arctan2(
        np.sqrt(2.0) * z * horizontal,
        1.0 - radius_squared,
    )
    latitude = np.arcsin(
        np.clip(np.sqrt(2.0) * z * vertical, -1.0, 1.0)
    )
    return longitude, latitude


def _hammer_page_coordinates(points, view_box):
    """Map SVG points to normalized Hammer coordinates."""
    x0, y0, width, height = view_box
    horizontal = 2.0 * (points[..., 0] - x0) / width - 1.0
    vertical = 1.0 - 2.0 * (points[..., 1] - y0) / height
    return horizontal, vertical


def _hammer_display_points(points, view_box):
    """Map SVG points to displayed Cartesian points on the unit sphere."""
    horizontal, vertical = _hammer_page_coordinates(points, view_box)
    longitude, latitude = _inverse_hammer_coordinates(horizontal, vertical)
    cosine_latitude = np.cos(latitude)
    return np.column_stack(
        (
            cosine_latitude * np.cos(longitude),
            cosine_latitude * np.sin(longitude),
            np.sin(latitude),
        )
    )


def _hammer_svg_sampling_coordinates(construction_N, view_box):
    """Return physical sphere and SVG coordinates for QuFlow grid samples."""
    import quflow as qf

    theta, phi = qf.sphgrid(construction_N)
    longitude = phi - np.pi
    latitude = theta - np.pi / 2.0
    horizontal, vertical = _hammer_coordinates(longitude, latitude)

    x0, y0, width, height = view_box
    svg_x = x0 + (horizontal + 1.0) * width / 2.0
    svg_y = y0 + (1.0 - vertical) * height / 2.0
    sphere_points = np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )
    svg_points = np.stack((svg_x, svg_y), axis=-1)
    return sphere_points, svg_points


def _curve_pixel_coordinates(curve, view_box, raster_shape):
    """Convert root-viewBox curve coordinates to raster coordinates."""
    x0, y0, width, height = view_box
    raster_height, raster_width = raster_shape
    column = (curve[:, 0] - x0) * (raster_width - 1) / width
    row = (curve[:, 1] - y0) * (raster_height - 1) / height
    return row, column


def _densify_polyline(curve, maximum_step):
    """Sample every polyline segment at no more than *maximum_step*."""
    samples = []
    for start, end in zip(curve[:-1], curve[1:]):
        step_count = max(
            1,
            int(math.ceil(np.linalg.norm(end - start) / maximum_step)),
        )
        fractions = np.arange(step_count, dtype=float) / step_count
        samples.append(start + fractions[:, np.newaxis] * (end - start))
    return np.concatenate(samples)


def _densify_hammer_curve(curve, view_box, maximum_step):
    """Densify by distance in the normalized Hammer display plane."""
    horizontal, vertical = _hammer_page_coordinates(curve, view_box)
    display_curve = np.column_stack((2.0 * horizontal, vertical))
    samples = []
    for index, (start, end) in enumerate(zip(curve[:-1], curve[1:])):
        display_step = display_curve[index + 1] - display_curve[index]
        step_count = max(
            1,
            int(math.ceil(np.linalg.norm(display_step) / maximum_step)),
        )
        fractions = np.arange(step_count, dtype=float) / step_count
        samples.append(start + fractions[:, np.newaxis] * (end - start))
    return np.concatenate(samples)


def _stereographic_curve_sphere_points(
    curve,
    view_box,
    chart_scale,
    basis,
    sampling_step,
):
    """Map flattened SVG curve points through the stereographic chart."""
    curve = _densify_polyline(curve, sampling_step)
    x0, y0, width, height = view_box
    u = chart_scale * 2.0 * (curve[:, 0] - (x0 + width / 2.0)) / width
    v = -chart_scale * 2.0 * (curve[:, 1] - (y0 + height / 2.0)) / height
    radius_squared = u**2 + v**2
    denominator = 1.0 + radius_squared
    local = np.column_stack(
        (
            2.0 * u / denominator,
            2.0 * v / denominator,
            (1.0 - radius_squared) / denominator,
        )
    )
    return local @ basis.T


def _stereographic_display_points(points, view_box, chart_scale, center):
    """Map SVG points through the displayed stereographic chart."""
    x0, y0, width, height = view_box
    u = chart_scale * 2.0 * (
        points[:, 0] - (x0 + width / 2.0)
    ) / width
    v = -chart_scale * 2.0 * (
        points[:, 1] - (y0 + height / 2.0)
    ) / height
    radius_squared = u**2 + v**2
    denominator = 1.0 + radius_squared
    local = np.column_stack(
        (
            2.0 * u / denominator,
            2.0 * v / denominator,
            (1.0 - radius_squared) / denominator,
        )
    )
    return local @ _chart_basis(center).T


def _blob_positions(displayed_points):
    """Convert displayed Cartesian points to ``qf.dynamics.blob`` points."""
    # qf.plot displays the physical grid antipodally, while blob's rotation
    # convention has the opposite y orientation.  Together these give
    # displayed (x, y, z) -> blob position (-x, +y, -z).
    return displayed_points * np.array((-1.0, 1.0, -1.0))


def _hammer_curve_geometry(curve, view_box, sampling_step):
    """Return dense displayed and physical points for a Hammer SVG curve."""
    curve = _densify_hammer_curve(curve, view_box, sampling_step)
    displayed_points = _hammer_display_points(curve, view_box)
    # qf.plot displays the antipode of the physical QuFlow grid point.
    return displayed_points, -displayed_points


def _ensure_hammer_margin(
    curves,
    view_box,
    sampling_step,
    sampling_padding_radians,
):
    """Keep every curve clear of the Hammer seam at sampling resolution."""
    required_clearance = sampling_padding_radians
    for curve_index, curve in enumerate(curves, start=1):
        dense_curve = _densify_hammer_curve(curve, view_box, sampling_step)
        horizontal, vertical = _hammer_page_coordinates(dense_curve, view_box)
        radius_squared = horizontal**2 + vertical**2
        if np.any(radius_squared >= 1.0):
            raise ValueError(
                f"SVG curve {curve_index} must lie strictly inside the "
                "Hammer ellipse; the rectangular page corners are outside "
                "the sphere."
            )

        longitude, latitude = _inverse_hammer_coordinates(
            horizontal,
            vertical,
        )
        cosine_latitude = np.cos(latitude)
        display_x = cosine_latitude * np.cos(longitude)
        display_y = cosine_latitude * np.sin(longitude)
        display_z = np.sin(latitude)

        # The Hammer boundary is the displayed antimeridian, with the poles
        # as its endpoints.  This is the exact spherical distance to that
        # half-great-circle seam.
        seam_clearance = np.where(
            display_x <= 0.0,
            np.arcsin(np.clip(np.abs(display_y), 0.0, 1.0)),
            np.arccos(np.clip(np.abs(display_z), 0.0, 1.0)),
        )
        minimum_clearance = float(np.min(seam_clearance))
        if minimum_clearance < required_clearance:
            raise ValueError(
                f"SVG curve {curve_index} is too close to the Hammer "
                "boundary for reliable sampling. Move it inward or increase "
                "construction_N."
            )


def _signed_spherical_distance_profile(
    inside,
    sphere_points,
    source_curve_points,
    distance_scale_radians,
):
    """Return signed angular distance, positive inside, in scale units."""
    inside = np.asarray(inside, dtype=bool).reshape(
        sphere_points.shape[:-1]
    )
    if not np.any(inside) or np.all(inside):
        raise ValueError(
            "An SVG curve has no resolvable interior. Increase "
            "construction_N or simplify/enlarge the curve."
        )

    chord_distance, _ = cKDTree(source_curve_points).query(
        sphere_points.reshape(-1, 3)
    )
    angular_distance = 2.0 * np.arcsin(
        np.clip(chord_distance / 2.0, 0.0, 1.0)
    ).reshape(sphere_points.shape[:-1])
    signed_distance = np.where(
        inside,
        angular_distance,
        -angular_distance,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return signed_distance / distance_scale_radians


def _hammer_curve_profile(
    curve,
    sphere_points,
    svg_points,
    source_curve_points,
    distance_scale_radians,
):
    """Sample one spherical signed-distance field on the QuFlow grid."""
    inside = MatplotlibPath(curve, closed=True).contains_points(
        svg_points.reshape(-1, 2)
    )
    return _signed_spherical_distance_profile(
        inside,
        sphere_points,
        source_curve_points,
        distance_scale_radians,
    )


def _ensure_stereographic_margin(
    curves,
    view_box,
    raster_shape,
):
    """Keep every curve inside the stereographic page at grid resolution."""
    raster_height, raster_width = raster_shape
    margin = 2.0
    for curve_index, curve in enumerate(curves, start=1):
        row, column = _curve_pixel_coordinates(curve, view_box, raster_shape)
        clearance = np.min(
            np.stack(
                (
                    row,
                    column,
                    raster_height - 1 - row,
                    raster_width - 1 - column,
                )
            )
        )
        if clearance < margin:
            raise ValueError(
                f"SVG curve {curve_index} is too close to the page edge for "
                "reliable sampling. Leave more margin, increase "
                "construction_N, or enlarge the page."
            )


def _sample_spherical_grid(field, theta, phi):
    """Sample a QuFlow MW grid periodically in longitude."""
    grid_N = field.shape[0]
    row = (theta * (2 * grid_N - 1) / np.pi - 1.0) / 2.0
    column = (phi % (2 * np.pi)) * (2 * grid_N - 1) / (2 * np.pi)
    return map_coordinates(
        field,
        (row, column),
        order=1,
        mode="grid-wrap",
    )


def _covariant_symbol(coefficients, matrix_size, validation_N):
    """Return the final Berezin symbol represented by a matrix's coefficients."""
    import quflow as qf

    berezin = qf.berezin_multipliers(matrix_size)
    return qf.shc2fun(
        coefficients * berezin,
        N=validation_N,
        isreal=True,
        berezin=False,
    )


def _validate_regular_symbol(
    symbol,
    level,
    validation_N,
    regularity_tolerance,
    *,
    source_curve_points,
    curve_tolerance_radians,
):
    """Numerically check the requested level of the final covariant symbol."""
    try:
        import contourpy
    except ImportError as error:
        raise ImportError(
            "SVG-domain validation requires contourpy; install contourpy or "
            "call import_svg_domain(..., validate=False)."
        ) from error

    import quflow as qf

    if not np.isfinite(symbol).all() or not (
        np.min(symbol) < level < np.max(symbol)
    ):
        raise ValueError(
            f"Level {level:g} disappeared after quantization. "
            "Increase N, "
            "enlarge or simplify the SVG curve, or reduce the collar width."
        )

    theta, phi = qf.sphgrid(validation_N)
    longitude = np.r_[phi[0], 2.0 * np.pi]
    periodic_symbol = np.column_stack((symbol, symbol[:, 0]))
    lines = contourpy.contour_generator(
        x=longitude,
        y=theta[:, 0],
        z=periodic_symbol,
    ).lines(level)
    if not lines:
        raise ValueError(
            f"Could not extract level {level:g} after quantization. "
            "Increase N or simplify the SVG curve."
        )

    contour_lines = []
    for line in lines:
        if line.shape[0] < 2:
            continue
        line_phi = line[:, 0] % (2.0 * np.pi)
        line_theta = line[:, 1]
        contour_lines.append(
            (
                line,
                np.column_stack(
                    (
                        np.sin(line_theta) * np.cos(line_phi),
                        np.sin(line_theta) * np.sin(line_phi),
                        np.cos(line_theta),
                    )
                ),
            )
        )
    if not contour_lines:
        raise ValueError(
            f"Level {level:g} has no nondegenerate contour after quantization."
        )

    # Keep the contour pieces which follow this particular source curve.  A
    # valid component can be split at longitude zero, hence several pieces may
    # be selected.  Other disconnected components of the same scalar level are
    # ignored for source-curve correspondence because band-limiting can create
    # additional disconnected contours.  Their gradients are still checked
    # below.
    source_tree = cKDTree(source_curve_points)
    chord_tolerance = 2.0 * np.sin(curve_tolerance_radians / 2.0)
    matching_points = []
    contour_to_source = []
    for line, points in contour_lines:
        chord_distance, _ = source_tree.query(points)
        if float(np.max(chord_distance)) <= chord_tolerance:
            matching_points.append(points)
            contour_to_source.append(chord_distance)

    if not matching_points:
        raise ValueError(
            f"Level {level:g} no longer follows its SVG curve after "
            "quantization. Increase N or simplify/enlarge the curve."
        )

    matching_points = np.concatenate(matching_points)
    source_to_contour, _ = cKDTree(matching_points).query(source_curve_points)
    maximum_chord_distance = max(
        float(np.max(source_to_contour)),
        max(float(np.max(distance)) for distance in contour_to_source),
    )
    maximum_distance = 2.0 * np.arcsin(
        np.clip(maximum_chord_distance / 2.0, 0.0, 1.0)
    )
    if maximum_distance > curve_tolerance_radians:
        raise ValueError(
            f"Level {level:g} moved {np.rad2deg(maximum_distance):.3g} "
            "degrees away from its SVG curve after quantization, exceeding "
            f"the {np.rad2deg(curve_tolerance_radians):.3g}-degree tolerance. "
            "Increase N or simplify/enlarge the curve."
        )

    spacing = 2.0 * np.pi / (2 * validation_N - 1)
    dtheta = np.gradient(symbol, spacing, axis=0, edge_order=2)
    dphi = (
        np.roll(symbol, -1, axis=1)
        - np.roll(symbol, 1, axis=1)
    ) / (2.0 * spacing)

    minimum_gradient = np.inf
    for line, _ in contour_lines:
        line_phi = line[:, 0] % (2.0 * np.pi)
        line_theta = line[:, 1]
        theta_derivative = _sample_spherical_grid(
            dtheta, line_theta, line_phi
        )
        phi_derivative = _sample_spherical_grid(dphi, line_theta, line_phi)
        sin_theta = np.maximum(np.sin(line_theta), 1e-12)
        gradient = np.sqrt(
            theta_derivative**2 + (phi_derivative / sin_theta) ** 2
        )
        if gradient.size:
            minimum_gradient = min(
                minimum_gradient,
                float(np.min(gradient)),
            )

    if (
        not np.isfinite(minimum_gradient)
        or minimum_gradient <= regularity_tolerance
    ):
        raise ValueError(
            f"Level {level:g} is not numerically regular after "
            f"quantization: minimum sampled spherical-gradient norm is "
            f"{minimum_gradient:.3e}. Increase N, round/simplify the SVG "
            "curve, or adjust collar_width_degrees."
        )


def import_svg_blobs(
    svg_file,
    N=128,
    *,
    sigma=0.0,
    mapping="hammer",
    center=(0.0, 0.0),
    chart_radius_degrees=60.0,
):
    """Create one QuFlow blob at each native Inkscape spiral centre.

    Parameters
    ----------
    svg_file : path-like
        An SVG containing native Inkscape spirals.  The spirals are centre
        annotations: their radius, turns, stroke, and path shape are ignored.
        Do not convert them to ordinary paths before saving the file.
    N : int, default=128
        Size of every returned QuFlow matrix.
    sigma : float, default=0
        Nonnegative Gaussian width passed to :func:`quflow.dynamics.blob`.
        The same width is used for every spiral.
    mapping : {"hammer", "stereographic"}, default="hammer"
        Mapping from the SVG page to the displayed sphere.  It has the same
        meaning as in :func:`import_svg_domain`.
    center : pair of float, default=(0, 0)
        Displayed latitude and longitude of the page centre in degrees.  Only
        available with stereographic mapping.
    chart_radius_degrees : float, default=60
        Stereographic distance from the page centre to an edge midpoint.  Only
        available with stereographic mapping.

    Returns
    -------
    list of ndarray
        One skew-Hermitian ``N`` by ``N`` blob matrix per visible native
        Inkscape spiral, in SVG document order.

    Notes
    -----
    With the default Hammer mapping, the SVG page is normalized to the full
    Hammer footprint used by :func:`quflow.plot`: the page centre maps to the
    plot centre, and SVG ``y`` is flipped so that the page top remains north.
    Spiral centres must lie inside the Hammer ellipse.  A 2:1 page gives the
    same visual proportions as the plot.
    """
    N = _positive_integer(N, "N")
    if N < 2:
        raise ValueError("N must be at least 2 for SVG blobs.")
    sigma = _finite_scalar(sigma, "sigma")
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative.")
    center_coordinates, chart_radius_degrees = _mapping_options(
        mapping,
        center,
        chart_radius_degrees,
    )

    try:
        svg_path = os.fspath(svg_file)
    except TypeError as error:
        raise TypeError("svg_file must be a filesystem path.") from error

    view_box, spiral_centers = _read_svg_spiral_centers(svg_path)
    x0, y0, width, height = view_box
    if mapping == "hammer":
        displayed_points = _hammer_display_points(
            spiral_centers,
            view_box,
        )
    else:
        aspect_tolerance = 1e-12 * max(width, height)
        if abs(width - height) > aspect_tolerance:
            raise ValueError(
                "Stereographic mapping requires a square SVG viewBox. Use, "
                "for example, viewBox='0 0 1000 1000'."
            )
        coordinate_tolerance = 64.0 * np.finfo(float).eps * max(
            width,
            height,
        )
        inside_page = (
            (spiral_centers[:, 0] >= x0 - coordinate_tolerance)
            & (spiral_centers[:, 0] <= x0 + width + coordinate_tolerance)
            & (spiral_centers[:, 1] >= y0 - coordinate_tolerance)
            & (spiral_centers[:, 1] <= y0 + height + coordinate_tolerance)
        )
        if not np.all(inside_page):
            raise ValueError(
                "SVG spiral centers must lie inside the root viewBox."
            )
        chart_scale = math.tan(
            math.radians(chart_radius_degrees) / 2.0
        )
        displayed_points = _stereographic_display_points(
            spiral_centers,
            view_box,
            chart_scale,
            center_coordinates,
        )

    import quflow as qf

    return [
        qf.dynamics.blob(N, position, sigma)
        for position in _blob_positions(displayed_points)
    ]


def import_svg_domain(
    svg_file,
    N=128,
    *,
    construction_N=None,
    mapping="hammer",
    center=(0.0, 0.0),
    chart_radius_degrees=60.0,
    collar_width_degrees=None,
    amplitude=0.45,
    validate=True,
    validation_N=None,
    regularity_tolerance=1e-6,
):
    r"""Import closed SVG curves as constraint-domain defining functions.

    Parameters
    ----------
    svg_file : path-like
        An Inkscape-compatible SVG.  For the default Hammer mapping, a 2:1
        ``viewBox`` such as ``0 0 2000 1000`` gives the same proportions as the
        plot, but other aspect ratios are normalized to the full footprint.
        Every visible closed path or closed subpath is one boundary.  Curves
        are returned in document order.  Convert text and clones to paths and
        leave a clear margin around every curve.  Keep Inkscape SVG metadata
        when the same file also contains spiral markers for
        :func:`import_svg_blobs`.
    N : int, default=128
        Size of the returned QuFlow matrix.
    construction_N : int, optional
        Bandwidth used while sampling the SVG.  The default is
        ``max(512, N)`` and it must be at least ``N``.
    mapping : {"hammer", "stereographic"}, default="hammer"
        ``"hammer"`` maps the SVG page directly to the complete footprint drawn
        by ``quflow.plot(..., projection="hammer")``.  The legacy
        ``"stereographic"`` mode maps a square page to a local spherical chart.
    center : pair of float, default=(0, 0)
        Displayed latitude and longitude, in degrees, of the SVG page centre
        in :func:`quflow.plot`.  Only available with stereographic mapping.
    chart_radius_degrees : float, default=60
        Spherical distance from the page centre to an edge midpoint.  The page
        is interpreted as a stereographic chart and must remain below 90
        degrees.  Only available with stereographic mapping.
    collar_width_degrees : float, optional
        Vertical distance scale of the defining field: moving this angular
        distance normally changes the field by ``amplitude``.  It controls the
        spacing in value between geometrically parallel contours but does not
        truncate the signed-distance profile.  The default is the matrix
        resolution ``degrees(1/sqrt(N))``.
    amplitude : float, default=0.45
        Positive vertical scale multiplying the signed-distance field.
    validate : bool, default=True
        Numerically check that every requested level survives truncation and
        has nonzero spherical gradient in the final Berezin symbol.
    validation_N : int, optional
        Grid bandwidth used for validation.  Defaults to ``construction_N``.
    regularity_tolerance : float, default=1e-6
        Minimum accepted sampled spherical-gradient norm.

    Returns
    -------
    functions : list of ndarray
        One skew-Hermitian ``N`` by ``N`` defining matrix per SVG curve.
    level_sets : list of float
        One ``0.0`` per matrix, in SVG document/subpath order.

    Notes
    -----
    The intended use is::

        functions, level_sets = import_svg_domain("domain.svg", N=128)
        F_constraint, actual = constraint_matrix(functions, level_sets)

    A regular level of a smooth function cannot have endpoints, corners,
    cusps, or self-intersections.  Each SVG curve is therefore treated as a
    smooth closed outline; finite bandwidth necessarily rounds sharp features.
    Different SVG curves may touch, cross, or nest because they are returned as
    independent functions.  Each field is signed distance: it is negative and
    continues descending outside its curve, and is positive and continues
    rising inside.  Levels on both sides of zero are therefore approximately
    parallel offsets, forming a mountain above the domain and descending
    terrain outside it.  Offset curves eventually meet at medial axes or cut
    loci; finite harmonic bandwidth rounds those nonsmooth ridges.  Validation
    checks the component corresponding to each SVG curve and checks regularity
    on all components of level ``0.0``.  It is a sampled numerical certificate,
    not a symbolic proof.

    In Hammer mode the page centre maps to the plot centre, its right-edge
    midpoint maps to the right tip, and its top midpoint maps to the north-pole
    tip.  The rectangular page corners are outside the Hammer ellipse and may
    not contain curves.  QuFlow's plotting-grid reversal is incorporated so
    the SVG's top remains visually up.
    """
    N = _positive_integer(N, "N")
    if N < 2:
        raise ValueError("N must be at least 2 for an SVG domain.")
    if construction_N is None:
        construction_N = max(512, N)
    construction_N = _positive_integer(construction_N, "construction_N")
    if construction_N < N:
        raise ValueError("construction_N must be greater than or equal to N.")

    center_coordinates, chart_radius_degrees = _mapping_options(
        mapping,
        center,
        chart_radius_degrees,
    )
    if collar_width_degrees is None:
        collar_width_degrees = math.degrees(1.0 / math.sqrt(N))
    collar_width_degrees = _finite_scalar(
        collar_width_degrees, "collar_width_degrees"
    )
    if collar_width_degrees <= 0.0:
        raise ValueError("collar_width_degrees must be positive.")
    amplitude = _finite_scalar(amplitude, "amplitude")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive.")
    regularity_tolerance = _finite_scalar(
        regularity_tolerance, "regularity_tolerance"
    )
    if regularity_tolerance < 0.0:
        raise ValueError("regularity_tolerance must be nonnegative.")
    if not isinstance(validate, (bool, np.bool_)):
        raise ValueError("validate must be a boolean.")
    if validation_N is None:
        validation_N = construction_N
    validation_N = _positive_integer(validation_N, "validation_N")
    if validation_N < N:
        raise ValueError("validation_N must be greater than or equal to N.")

    try:
        svg_path = os.fspath(svg_file)
    except TypeError as error:
        raise TypeError("svg_file must be a filesystem path.") from error

    # Flatten to substantially less than one raster pixel in root coordinates.
    view_box, curves = _read_svg_curves(svg_path)
    x0, y0, width, height = view_box
    aspect_tolerance = 1e-12 * max(width, height)
    raster_size = 2 * construction_N - 1
    distance_scale_radians = math.radians(collar_width_degrees)
    if distance_scale_radians == 0.0:
        raise ValueError(
            "collar_width_degrees is too small to represent in radians."
        )

    if mapping == "hammer":
        angular_pixel_size = 2.0 * math.sqrt(2.0) / (raster_size - 1)
        sampling_step = 2.0 / (raster_size - 1)
    else:
        if abs(width - height) > aspect_tolerance:
            raise ValueError(
                "Stereographic mapping requires a square SVG viewBox. Use, "
                "for example, viewBox='0 0 1000 1000'."
            )
        raster_shape = (raster_size, raster_size)
        chart_scale = math.tan(math.radians(chart_radius_degrees) / 2.0)
        sampling_step = max(width, height) / (raster_size - 1)

    if mapping == "hammer":
        _ensure_hammer_margin(
            curves,
            view_box,
            sampling_step,
            2.0 * angular_pixel_size,
        )
        sphere_points, svg_points = _hammer_svg_sampling_coordinates(
            construction_N,
            view_box,
        )
    else:
        _ensure_stereographic_margin(
            curves,
            view_box,
            raster_shape,
        )

        # qf.plot displays phi=0 at -180 degrees and reverses physical
        # latitude.  The antipodal basis makes `center` refer to displayed
        # plot coordinates and keeps the SVG vertically upright.
        basis = -_chart_basis(center_coordinates)
        (
            sphere_points,
            sample_row,
            sample_column,
            valid,
        ) = _stereographic_sampling_coordinates(
            construction_N,
            raster_size,
            chart_scale,
            basis,
        )
        valid_svg_points = np.column_stack(
            (
                x0
                + sample_column[valid] * width / (raster_size - 1),
                y0 + sample_row[valid] * height / (raster_size - 1),
            )
        )

    import quflow as qf

    functions = []
    level_sets = [0.0] * len(curves)
    for curve_index, curve in enumerate(curves, start=1):
        if mapping == "hammer":
            _, source_curve_points = _hammer_curve_geometry(
                curve,
                view_box,
                sampling_step,
            )
            profile = _hammer_curve_profile(
                curve,
                sphere_points,
                svg_points,
                source_curve_points,
                distance_scale_radians,
            )
        else:
            inside = np.zeros(sphere_points.shape[:-1], dtype=bool)
            inside[valid] = MatplotlibPath(
                curve,
                closed=True,
            ).contains_points(valid_svg_points)
            source_curve_points = _stereographic_curve_sphere_points(
                curve,
                view_box,
                chart_scale,
                basis,
                sampling_step,
            )
            profile = _signed_spherical_distance_profile(
                inside,
                sphere_points,
                source_curve_points,
                distance_scale_radians,
            )
        with np.errstate(over="ignore", invalid="ignore"):
            centered_field = amplitude * profile
        if not np.isfinite(centered_field).all():
            raise ValueError(
                "amplitude and collar_width_degrees must produce finite "
                "signed-distance values."
            )
        coefficients = qf.fun2shc(centered_field)[: N**2]
        function = qf.shc2mat(coefficients, N=N, berezin=False)
        if not np.isfinite(function).all():
            raise ValueError(
                f"SVG curve {curve_index} produced non-finite matrix entries."
            )
        # Remove transform roundoff without changing the represented real
        # scalar field to working precision.
        function = 0.5 * (function - function.conj().T)

        if validate:
            eigenvalues = np.linalg.eigvalsh(-1j * function)
            if not eigenvalues[0] < 0.0 < eigenvalues[-1]:
                raise ValueError(
                    f"Level 0 for SVG curve {curve_index} does not lie inside "
                    f"the matrix spectrum [{eigenvalues[0]:.6g}, "
                    f"{eigenvalues[-1]:.6g}]. Increase N or enlarge/simplify "
                    "the SVG curve."
                )

            symbol = _covariant_symbol(coefficients, N, validation_N)
            curve_tolerance_radians = 2.0 / math.sqrt(N)
            if mapping == "hammer":
                validation_step = 1.0 / (validation_N - 1)
                _, validation_curve_points = _hammer_curve_geometry(
                    curve,
                    view_box,
                    validation_step,
                )
            else:
                validation_curve_points = _stereographic_curve_sphere_points(
                    curve,
                    view_box,
                    chart_scale,
                    basis,
                    max(width, height) / (2 * validation_N - 1),
                )

            _validate_regular_symbol(
                symbol,
                0.0,
                validation_N,
                regularity_tolerance,
                source_curve_points=validation_curve_points,
                curve_tolerance_radians=curve_tolerance_radians,
            )
        functions.append(function)

    return functions, level_sets
