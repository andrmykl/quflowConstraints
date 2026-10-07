"""Small SVG-to-polyline reader used by the constraint-domain importer.

This module deliberately implements only the geometry needed for authoring
constraint boundaries.  It is not an SVG renderer: paint, clipping, reusable
elements and physical units are outside its scope.  Curves are returned in
root ``viewBox`` coordinates after applying SVG transforms.

Path tokenization and elliptical-arc conversion are delegated to FontTools.
The dependency is imported lazily so importing :mod:`quflow` does not require
FontTools unless an SVG domain is actually read.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import numpy as np


__all__ = ["_read_svg_curves", "_read_svg_spiral_centers"]


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER_PATTERN)
_SINGLE_NUMBER_RE = re.compile(rf"\s*({_NUMBER_PATTERN})\s*\Z")
_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^()]*)\)")
_SEPARATORS_RE = re.compile(r"[\s,]*\Z")

_SODIPODI_NAMESPACE = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
_SODIPODI_TYPE = f"{{{_SODIPODI_NAMESPACE}}}type"
_SODIPODI_CX = f"{{{_SODIPODI_NAMESPACE}}}cx"
_SODIPODI_CY = f"{{{_SODIPODI_NAMESPACE}}}cy"

_GEOMETRY_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "polyline"}
_UNSUPPORTED_GRAPHICS = {
    "image",
    "line",
    "mesh",
    "text",
    "use",
}
_NON_RENDERED_CONTAINERS = {
    "clipPath",
    "defs",
    "marker",
    "mask",
    "metadata",
    "pattern",
    "symbol",
}


class _SafeTreeBuilder(ET.TreeBuilder):
    """Reject document types and entity declarations not needed by plain SVG."""

    def doctype(self, name, public_id, system_id):
        raise ValueError(
            "SVG document type and entity declarations are not supported."
        )


def _local_name(tag):
    """Return an XML tag without its namespace."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _parse_number_list(value, description):
    """Parse a comma/whitespace-separated SVG number list."""
    if value is None:
        raise ValueError(f"Missing {description}.")

    matches = list(_NUMBER_RE.finditer(value))
    if not matches:
        raise ValueError(f"{description} must contain finite numbers.")

    end = 0
    numbers = []
    for match in matches:
        if not _SEPARATORS_RE.fullmatch(value[end : match.start()]):
            raise ValueError(f"Invalid {description}: {value!r}.")
        number = float(match.group())
        if not math.isfinite(number):
            raise ValueError(f"{description} must contain finite numbers.")
        numbers.append(number)
        end = match.end()
    if not _SEPARATORS_RE.fullmatch(value[end:]):
        raise ValueError(f"Invalid {description}: {value!r}.")
    return numbers


def _number_attribute(element, name, *, default=None):
    """Read one unitless numeric SVG attribute."""
    value = element.get(name)
    if value is None:
        if default is not None:
            return float(default)
        raise ValueError(
            f"<{_local_name(element.tag)}> is missing its {name!r} attribute."
        )
    match = _SINGLE_NUMBER_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Attribute {name!r} on <{_local_name(element.tag)}> must be a "
            "unitless finite number in viewBox coordinates."
        )
    result = float(match.group(1))
    if not math.isfinite(result):
        raise ValueError(
            f"Attribute {name!r} on <{_local_name(element.tag)}> must be finite."
        )
    return result


def _translation(tx, ty=0.0):
    return np.array(
        ((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0)),
        dtype=float,
    )


def _parse_transform(value):
    """Return the 3-by-3 affine matrix for an SVG transform list."""
    if value is None or not value.strip():
        return np.eye(3)

    transform = np.eye(3)
    end = 0
    found = False
    for match in _TRANSFORM_RE.finditer(value):
        if not _SEPARATORS_RE.fullmatch(value[end : match.start()]):
            raise ValueError(f"Invalid SVG transform: {value!r}.")
        found = True
        name = match.group(1)
        arguments = _parse_number_list(
            match.group(2), f"arguments to transform {name!r}"
        )

        if name == "matrix" and len(arguments) == 6:
            a, b, c, d, e, f = arguments
            operation = np.array(
                ((a, c, e), (b, d, f), (0.0, 0.0, 1.0)), dtype=float
            )
        elif name == "translate" and len(arguments) in (1, 2):
            operation = _translation(
                arguments[0], arguments[1] if len(arguments) == 2 else 0.0
            )
        elif name == "scale" and len(arguments) in (1, 2):
            sx = arguments[0]
            sy = arguments[1] if len(arguments) == 2 else sx
            operation = np.diag((sx, sy, 1.0))
        elif name == "rotate" and len(arguments) in (1, 3):
            angle = math.radians(arguments[0])
            cosine = math.cos(angle)
            sine = math.sin(angle)
            rotation = np.array(
                (
                    (cosine, -sine, 0.0),
                    (sine, cosine, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=float,
            )
            if len(arguments) == 3:
                cx, cy = arguments[1:]
                operation = (
                    _translation(cx, cy)
                    @ rotation
                    @ _translation(-cx, -cy)
                )
            else:
                operation = rotation
        elif name == "skewX" and len(arguments) == 1:
            tangent = math.tan(math.radians(arguments[0]))
            operation = np.array(
                ((1.0, tangent, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=float,
            )
        elif name == "skewY" and len(arguments) == 1:
            tangent = math.tan(math.radians(arguments[0]))
            operation = np.array(
                ((1.0, 0.0, 0.0), (tangent, 1.0, 0.0), (0.0, 0.0, 1.0)),
                dtype=float,
            )
        else:
            raise ValueError(
                f"Unsupported or malformed SVG transform {name!r} in {value!r}."
            )

        if not np.isfinite(operation).all():
            raise ValueError(f"SVG transform must be finite: {value!r}.")

        # SVG transform lists are equivalent to nested transforms in the
        # order written.  For column vectors that gives op1 @ op2 @ ... .
        transform = transform @ operation
        end = match.end()

    if not found or not _SEPARATORS_RE.fullmatch(value[end:]):
        raise ValueError(f"Invalid SVG transform: {value!r}.")
    return transform


def _style_properties(element):
    properties = {}
    for declaration in element.get("style", "").split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", 1)
        properties[name.strip()] = value.strip()
    return properties


class _SVGPathGeometryError(ValueError):
    """A valid path command stream which violates the domain contract."""


def _fonttools_parse_path(path_data, pen):
    """Parse path data with a lazy, actionable FontTools import."""
    try:
        from fontTools.svgLib.path import parse_path
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "Reading SVG domain paths requires FontTools. Install it with "
            "`python -m pip install fonttools`."
        ) from error

    try:
        parse_path(path_data, pen)
    except _SVGPathGeometryError:
        raise
    except (AssertionError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid SVG path data: {path_data!r}.") from error


def _point_line_distance(point, start, end):
    chord = end - start
    length = np.linalg.norm(chord)
    if length == 0.0:
        return float(np.linalg.norm(point - start))
    cross_product = chord[0] * (point[1] - start[1]) - chord[1] * (
        point[0] - start[0]
    )
    return float(abs(cross_product) / length)


class _FlatteningPen:
    """A minimal FontTools pen which adaptively flattens closed subpaths."""

    _MAX_DEPTH = 24
    _MAX_POINTS = 10_000

    def __init__(self, transform, flatness):
        self.transform = np.asarray(transform, dtype=float)
        self.flatness = float(flatness)
        self.curves = []
        self._points = None

    def _transform_point(self, point):
        point = np.asarray((point[0], point[1], 1.0), dtype=float)
        transformed = self.transform @ point
        return transformed[:2]

    def _append(self, point):
        if len(self._points) >= self._MAX_POINTS:
            raise _SVGPathGeometryError(
                "SVG curve produced too many points; simplify the path or "
                "increase flatness."
            )
        self._points.append(np.asarray(point, dtype=float))

    def moveTo(self, point):
        if self._points is not None:
            raise _SVGPathGeometryError(
                "Every SVG path subpath must be explicitly closed."
            )
        self._points = [self._transform_point(point)]

    def lineTo(self, point):
        if self._points is None:
            raise _SVGPathGeometryError(
                "SVG path draws before its initial move command."
            )
        self._append(self._transform_point(point))

    def curveTo(self, *points):
        if self._points is None or len(points) != 3:
            raise _SVGPathGeometryError("Unsupported cubic SVG path segment.")
        start = self._points[-1]
        control1, control2, end = map(self._transform_point, points)
        self._flatten_cubic(start, control1, control2, end, 0)

    def qCurveTo(self, *points):
        if self._points is None or len(points) != 2 or points[-1] is None:
            raise _SVGPathGeometryError(
                "Unsupported quadratic SVG path segment."
            )
        start = self._points[-1]
        control, end = map(self._transform_point, points)
        self._flatten_quadratic(start, control, end, 0)

    def closePath(self):
        if self._points is None:
            raise _SVGPathGeometryError(
                "SVG close command has no open subpath."
            )
        if not np.array_equal(self._points[-1], self._points[0]):
            self._append(self._points[0].copy())
        self.curves.append(np.asarray(self._points, dtype=float))
        self._points = None

    def endPath(self):
        raise _SVGPathGeometryError(
            "Every SVG path subpath must be explicitly closed with Z."
        )

    def _flatten_quadratic(self, start, control, end, depth):
        control_net_excess = (
            np.linalg.norm(control - start)
            + np.linalg.norm(end - control)
            - np.linalg.norm(end - start)
        )
        if depth >= self._MAX_DEPTH or (
            _point_line_distance(control, start, end) <= self.flatness
            and control_net_excess <= self.flatness
        ):
            self._append(end)
            return

        start_control = 0.5 * (start + control)
        control_end = 0.5 * (control + end)
        midpoint = 0.5 * (start_control + control_end)
        self._flatten_quadratic(
            start, start_control, midpoint, depth + 1
        )
        self._flatten_quadratic(midpoint, control_end, end, depth + 1)

    def _flatten_cubic(self, start, control1, control2, end, depth):
        control_net_excess = (
            np.linalg.norm(control1 - start)
            + np.linalg.norm(control2 - control1)
            + np.linalg.norm(end - control2)
            - np.linalg.norm(end - start)
        )
        if depth >= self._MAX_DEPTH or (
            max(
                _point_line_distance(control1, start, end),
                _point_line_distance(control2, start, end),
            )
            <= self.flatness
            and control_net_excess <= self.flatness
        ):
            self._append(end)
            return

        point01 = 0.5 * (start + control1)
        point12 = 0.5 * (control1 + control2)
        point23 = 0.5 * (control2 + end)
        point012 = 0.5 * (point01 + point12)
        point123 = 0.5 * (point12 + point23)
        midpoint = 0.5 * (point012 + point123)
        self._flatten_cubic(
            start, point01, point012, midpoint, depth + 1
        )
        self._flatten_cubic(midpoint, point123, point23, end, depth + 1)


def _shape_path(element):
    """Return path data for one supported SVG geometry element."""
    tag = _local_name(element.tag)
    if tag == "path":
        path = element.get("d")
        if path is None or not path.strip():
            raise ValueError("<path> must have nonempty path data in 'd'.")
        return path

    if tag in ("circle", "ellipse"):
        cx = _number_attribute(element, "cx", default=0.0)
        cy = _number_attribute(element, "cy", default=0.0)
        if tag == "circle":
            rx = ry = _number_attribute(element, "r")
        else:
            rx = _number_attribute(element, "rx")
            ry = _number_attribute(element, "ry")
        if rx <= 0.0 or ry <= 0.0:
            raise ValueError(f"<{tag}> radii must be positive.")
        return (
            f"M {cx - rx},{cy} "
            f"A {rx},{ry} 0 1 0 {cx + rx},{cy} "
            f"A {rx},{ry} 0 1 0 {cx - rx},{cy} Z"
        )

    if tag == "rect":
        x = _number_attribute(element, "x", default=0.0)
        y = _number_attribute(element, "y", default=0.0)
        width = _number_attribute(element, "width")
        height = _number_attribute(element, "height")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("<rect> width and height must be positive.")

        rx_value = element.get("rx")
        ry_value = element.get("ry")
        rx = (
            _number_attribute(element, "rx")
            if rx_value is not None
            else None
        )
        ry = (
            _number_attribute(element, "ry")
            if ry_value is not None
            else None
        )
        if rx is None and ry is None:
            rx = ry = 0.0
        elif rx is None:
            rx = ry
        elif ry is None:
            ry = rx
        if rx < 0.0 or ry < 0.0:
            raise ValueError("<rect> corner radii cannot be negative.")
        rx = min(rx, width / 2.0)
        ry = min(ry, height / 2.0)
        if rx == 0.0 or ry == 0.0:
            return (
                f"M {x},{y} H {x + width} V {y + height} "
                f"H {x} Z"
            )
        return (
            f"M {x + rx},{y} H {x + width - rx} "
            f"A {rx},{ry} 0 0 1 {x + width},{y + ry} "
            f"V {y + height - ry} "
            f"A {rx},{ry} 0 0 1 {x + width - rx},{y + height} "
            f"H {x + rx} A {rx},{ry} 0 0 1 {x},{y + height - ry} "
            f"V {y + ry} A {rx},{ry} 0 0 1 {x + rx},{y} Z"
        )

    points = _parse_number_list(element.get("points"), f"<{tag}> points")
    if len(points) < 6 or len(points) % 2:
        raise ValueError(f"<{tag}> must contain at least three point pairs.")
    point_pairs = list(zip(points[::2], points[1::2]))
    if tag == "polyline" and point_pairs[-1] != point_pairs[0]:
        raise ValueError("<polyline> must repeat its first point to be closed.")
    coordinates = " ".join(f"{x},{y}" for x, y in point_pairs)
    return f"M {coordinates} Z"


def _element_description(element):
    tag = _local_name(element.tag)
    identifier = element.get("id")
    return f"<{tag} id={identifier!r}>" if identifier else f"<{tag}>"


def _is_inkscape_spiral(element):
    """Return whether *element* is a native Inkscape spiral marker."""
    return (
        _local_name(element.tag) == "path"
        and element.get(_SODIPODI_TYPE, "").strip() == "spiral"
    )


def _spiral_coordinate(element, attribute, display_name):
    """Read one finite, unitless native Inkscape spiral coordinate."""
    value = element.get(attribute)
    description = _element_description(element)
    if value is None:
        raise ValueError(
            f"Inkscape spiral {description} is missing {display_name}."
        )
    match = _SINGLE_NUMBER_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            f"Inkscape spiral {description} attribute {display_name} must "
            "be a unitless finite number in viewBox coordinates."
        )
    coordinate = float(match.group(1))
    if not math.isfinite(coordinate):
        raise ValueError(
            f"Inkscape spiral {description} attribute {display_name} must "
            "be finite."
        )
    return coordinate


def _transformed_spiral_center(element, transform):
    """Return an Inkscape spiral's transformed root-viewBox centre."""
    center = np.array(
        (
            _spiral_coordinate(element, _SODIPODI_CX, "sodipodi:cx"),
            _spiral_coordinate(element, _SODIPODI_CY, "sodipodi:cy"),
            1.0,
        )
    )
    center = transform @ center
    if not np.isfinite(center).all():
        raise ValueError(
            f"Inkscape spiral {_element_description(element)} has a "
            "non-finite transformed centre."
        )
    return center[:2]


def _collect_geometry(
    element,
    parent_transform,
    displayed,
    inherited_visibility,
    curves,
    flatness,
    *,
    spiral_centers=None,
    collect_curves=True,
    is_root=False,
):
    """Walk visible SVG elements in document order."""
    tag = _local_name(element.tag)
    if tag in _NON_RENDERED_CONTAINERS:
        return

    style = _style_properties(element)

    own_display = style.get("display", element.get("display", "inline"))
    displayed = displayed and own_display.strip().lower() != "none"
    if not displayed:
        return

    visibility = style.get(
        "visibility", element.get("visibility", inherited_visibility)
    ).strip().lower()
    if visibility == "inherit":
        visibility = inherited_visibility
    visible = visibility not in ("hidden", "collapse")

    for unsupported_property in ("clip-path", "mask"):
        property_value = style.get(
            unsupported_property, element.get(unsupported_property)
        )
        if (
            visible
            and property_value is not None
            and property_value.strip().lower() != "none"
        ):
            raise ValueError(
                f"{unsupported_property} on {_element_description(element)} "
                "is not supported; apply it and convert the result to a "
                "closed path in Inkscape."
            )

    local_transform = _parse_transform(element.get("transform"))
    transform = parent_transform @ local_transform
    if not np.isfinite(transform).all():
        raise ValueError(
            f"Transform on {_element_description(element)} is not finite."
        )

    if tag == "svg" and not is_root:
        raise ValueError("Nested <svg> viewports are not supported.")

    if visible and _is_inkscape_spiral(element):
        if spiral_centers is not None:
            spiral_centers.append(
                _transformed_spiral_center(element, transform)
            )
        # Native Inkscape spirals are point annotations for blob placement,
        # not closed domain boundaries.
        return

    if tag in _GEOMETRY_TAGS and visible:
        if not collect_curves:
            return
        pen = _FlatteningPen(transform, flatness)
        try:
            _fonttools_parse_path(_shape_path(element), pen)
        except (ImportError, ValueError) as error:
            error_type = type(error)
            raise error_type(
                f"Could not read {_element_description(element)}: {error}"
            ) from error
        curves.extend(pen.curves)
        return

    if tag in _UNSUPPORTED_GRAPHICS and visible:
        raise ValueError(
            f"Unsupported visible SVG element {_element_description(element)}. "
            "Convert it to a closed path in Inkscape."
        )

    # Groups, links and the root SVG are traversed.  Unknown foreign-namespace
    # metadata is harmless and is likewise traversed in case it wraps groups.
    for child in element:
        _collect_geometry(
            child,
            transform,
            displayed,
            visibility,
            curves,
            flatness,
            spiral_centers=spiral_centers,
            collect_curves=collect_curves,
        )


def _clean_curve(curve, tolerance):
    """Remove numerical duplicate vertices and validate nondegeneracy."""
    if curve.ndim != 2 or curve.shape[1] != 2 or not np.isfinite(curve).all():
        raise ValueError("SVG curves must contain finite two-dimensional points.")

    cleaned = [curve[0]]
    for point in curve[1:]:
        if np.linalg.norm(point - cleaned[-1]) > tolerance:
            cleaned.append(point)
    cleaned = np.asarray(cleaned, dtype=float)

    if np.linalg.norm(cleaned[-1] - cleaned[0]) <= tolerance:
        cleaned[-1] = cleaned[0]
    else:  # This should only be reachable through a broken pen implementation.
        raise ValueError("SVG curve is not closed.")

    if cleaned.shape[0] < 4:
        raise ValueError("SVG curve is degenerate; it needs three distinct points.")

    segments = np.diff(cleaned, axis=0)
    perimeter = float(np.linalg.norm(segments, axis=1).sum())
    if perimeter <= tolerance:
        raise ValueError("SVG curve is degenerate or has zero perimeter.")
    return cleaned


def _cross(first, second):
    return float(first[0] * second[1] - first[1] * second[0])


def _segments_intersect(first_start, first_end, second_start, second_end, tolerance):
    """Return whether two closed planar segments touch or cross."""
    if (
        max(first_start[0], first_end[0]) + tolerance
        < min(second_start[0], second_end[0])
        or max(second_start[0], second_end[0]) + tolerance
        < min(first_start[0], first_end[0])
        or max(first_start[1], first_end[1]) + tolerance
        < min(second_start[1], second_end[1])
        or max(second_start[1], second_end[1]) + tolerance
        < min(first_start[1], first_end[1])
    ):
        return False

    first_vector = first_end - first_start
    second_vector = second_end - second_start
    length_scale = max(
        np.linalg.norm(first_vector), np.linalg.norm(second_vector), 1.0
    )
    cross_tolerance = tolerance * length_scale
    orientations = (
        _cross(first_vector, second_start - first_start),
        _cross(first_vector, second_end - first_start),
        _cross(second_vector, first_start - second_start),
        _cross(second_vector, first_end - second_start),
    )

    def sign(value):
        if value > cross_tolerance:
            return 1
        if value < -cross_tolerance:
            return -1
        return 0

    signs = tuple(sign(value) for value in orientations)
    if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
        return True

    def on_segment(point, start, end, orientation):
        return (
            abs(orientation) <= cross_tolerance
            and min(start[0], end[0]) - tolerance
            <= point[0]
            <= max(start[0], end[0]) + tolerance
            and min(start[1], end[1]) - tolerance
            <= point[1]
            <= max(start[1], end[1]) + tolerance
        )

    return (
        on_segment(second_start, first_start, first_end, orientations[0])
        or on_segment(second_end, first_start, first_end, orientations[1])
        or on_segment(first_start, second_start, second_end, orientations[2])
        or on_segment(first_end, second_start, second_end, orientations[3])
    )


def _curve_self_intersects(curve, tolerance):
    segment_count = curve.shape[0] - 1
    for first in range(segment_count):
        first_start, first_end = curve[first : first + 2]
        for second in range(first + 1, segment_count):
            if second == first + 1 or (first == 0 and second == segment_count - 1):
                continue
            second_start, second_end = curve[second : second + 2]
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
                tolerance,
            ):
                return True
    return False


def _validate_topology(curves, tolerance):
    # The exact segment tests below are intentionally simple and reliable, but
    # quadratic.  Refuse pathological exported paths before they can tie up an
    # import for minutes.  Ordinary Inkscape paths are far below this budget.
    segment_counts = [curve.shape[0] - 1 for curve in curves]
    comparison_budget = sum(
        max(segment_count * (segment_count - 3) // 2, 0)
        for segment_count in segment_counts
    )
    if comparison_budget > 5_000_000:
        raise ValueError(
            "The SVG paths are too geometrically complex for safe topology "
            "validation. Simplify the paths in Inkscape and try again."
        )

    for index, curve in enumerate(curves, start=1):
        if _curve_self_intersects(curve, tolerance):
            raise ValueError(f"SVG curve {index} self-intersects or touches itself.")
        perimeter = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
        twice_area = abs(
            float(
                np.dot(curve[:-1, 0], curve[1:, 1])
                - np.dot(curve[:-1, 1], curve[1:, 0])
            )
        )
        area_tolerance = (
            128.0 * np.finfo(float).eps * max(perimeter**2, 1.0)
        )
        if twice_area <= area_tolerance:
            raise ValueError(
                f"SVG curve {index} is degenerate or has zero enclosed area."
            )


def _parse_svg_document(svg_file):
    """Return a safely parsed SVG root and its validated root viewBox."""
    try:
        parser = ET.XMLParser(target=_SafeTreeBuilder())
        root = ET.parse(svg_file, parser=parser).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Could not parse SVG file: {error}") from error
    except TypeError as error:
        raise ValueError(f"Could not parse SVG input: {error}") from error

    if _local_name(root.tag) != "svg":
        raise ValueError("The document root must be an <svg> element.")

    view_box_values = _parse_number_list(root.get("viewBox"), "root viewBox")
    if len(view_box_values) != 4:
        raise ValueError("The root viewBox must contain exactly four numbers.")
    x0, y0, width, height = view_box_values
    if width <= 0.0 or height <= 0.0:
        raise ValueError("The root viewBox width and height must be positive.")
    view_box = (x0, y0, width, height)
    return root, view_box


def _read_svg_curves(svg_file, *, flatness=None):
    """Read visible, closed SVG curves as flattened viewBox polylines.

    Native Inkscape spirals are reserved as blob-centre annotations and are
    ignored here.

    Parameters
    ----------
    svg_file:
        A path or binary/text file object accepted by
        :func:`xml.etree.ElementTree.parse`.
    flatness: float, optional
        Maximum Bézier-to-polyline error in root viewBox units.  The default is
        ``1e-4 * max(viewBox width, viewBox height)``.
    Returns
    -------
    view_box: tuple
        ``(x, y, width, height)`` from the required root ``viewBox``.
    curves: list of ndarray
        Closed ``(n, 2)`` float polylines, including the repeated endpoint, in
        SVG document and path-subpath order.
    """
    root, view_box = _parse_svg_document(svg_file)
    _, _, width, height = view_box

    scale = max(width, height)
    if flatness is None:
        flatness = scale * 1e-4
    flatness_array = np.asarray(flatness)
    if (
        flatness_array.ndim != 0
        or np.iscomplexobj(flatness_array)
        or flatness_array.dtype.kind not in "iuf"
    ):
        raise ValueError("flatness must be a finite positive real scalar.")
    flatness = float(flatness_array)
    if not math.isfinite(flatness) or flatness <= 0.0:
        raise ValueError("flatness must be a finite positive real scalar.")

    curves = []
    _collect_geometry(
        root,
        np.eye(3),
        True,
        "visible",
        curves,
        flatness,
        is_root=True,
    )
    if not curves:
        raise ValueError("The SVG contains no visible closed curves.")
    numerical_tolerance = max(
        scale * 64.0 * np.finfo(float).eps,
        flatness * 1e-10,
    )
    curves = [_clean_curve(curve, numerical_tolerance) for curve in curves]
    topology_tolerance = max(scale * 1e-10, numerical_tolerance)
    _validate_topology(curves, topology_tolerance)
    return view_box, curves


def _read_svg_spiral_centers(svg_file):
    """Read visible native Inkscape spiral centres in document order."""
    root, view_box = _parse_svg_document(svg_file)
    spiral_centers = []
    _collect_geometry(
        root,
        np.eye(3),
        True,
        "visible",
        [],
        None,
        spiral_centers=spiral_centers,
        collect_curves=False,
        is_root=True,
    )
    if not spiral_centers:
        raise ValueError("The SVG contains no visible native Inkscape spirals.")
    return view_box, np.asarray(spiral_centers, dtype=float)
