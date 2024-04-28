"""
This type stub file was generated by pyright.
"""

import enum
import functools

from . import _api

"""
Implementation details for :mod:`.mathtext`.
"""
_log = ...

@_api.delete_parameter("3.6", "math")
def get_unicode_index(symbol, math=...):  # -> int:
    r"""
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single (Unicode) character, a TeX command (e.g. r'\pi') or a Type1
        symbol name (e.g. 'phi').
    math : bool, default: False
        If True (deprecated), replace ASCII hyphen-minus by Unicode minus.
    """
    ...

VectorParse = ...
RasterParse = ...

class Output:
    r"""
    Result of `ship`\ping a box: lists of positioned glyphs and rectangles.

    This class is not exposed to end users, but converted to a `VectorParse` or
    a `RasterParse` by `.MathTextParser.parse`.
    """
    def __init__(self, box) -> None: ...
    def to_vector(self):  # -> VectorParse:
        ...
    def to_raster(self):  # -> RasterParse:
        ...

class Fonts:
    """
    An abstract base class for a system of fonts to use for mathtext.

    The class must be able to take symbol keys and font file names and
    return the character metrics.  It also delegates to a backend class
    to do the actual drawing.
    """

    def __init__(self, default_font_prop, load_glyph_flags) -> None:
        """
        Parameters
        ----------
        default_font_prop : `~.font_manager.FontProperties`
            The default non-math font, or the base font for Unicode (generic)
            font rendering.
        load_glyph_flags : int
            Flags passed to the glyph loader (e.g. ``FT_Load_Glyph`` and
            ``FT_Load_Char`` for FreeType-based fonts).
        """
        ...
    def get_kern(
        self,
        font1,
        fontclass1,
        sym1,
        fontsize1,
        font2,
        fontclass2,
        sym2,
        fontsize2,
        dpi,
    ):  # -> float:
        """
        Get the kerning distance for font between *sym1* and *sym2*.

        See `~.Fonts.get_metrics` for a detailed description of the parameters.
        """
        ...
    def get_metrics(self, font, font_class, sym, fontsize, dpi):
        r"""
        Parameters
        ----------
        font : str
            One of the TeX font names: "tt", "it", "rm", "cal", "sf", "bf",
            "default", "regular", "bb", "frak", "scr".  "default" and "regular"
            are synonyms and use the non-math font.
        font_class : str
            One of the TeX font names (as for *font*), but **not** "bb",
            "frak", or "scr".  This is used to combine two font classes.  The
            only supported combination currently is ``get_metrics("frak", "bf",
            ...)``.
        sym : str
            A symbol in raw TeX form, e.g., "1", "x", or "\sigma".
        fontsize : float
            Font size in points.
        dpi : float
            Rendering dots-per-inch.

        Returns
        -------
        object

            The returned object has the following attributes (all floats,
            except *slanted*):

            - *advance*: The advance distance (in points) of the glyph.
            - *height*: The height of the glyph in points.
            - *width*: The width of the glyph in points.
            - *xmin*, *xmax*, *ymin*, *ymax*: The ink rectangle of the glyph
            - *iceberg*: The distance from the baseline to the top of the
              glyph.  (This corresponds to TeX's definition of "height".)
            - *slanted*: Whether the glyph should be considered as "slanted"
              (currently used for kerning sub/superscripts).
        """
        ...
    def render_glyph(
        self, output, ox, oy, font, font_class, sym, fontsize, dpi
    ):  # -> None:
        """
        At position (*ox*, *oy*), draw the glyph specified by the remaining
        parameters (see `get_metrics` for their detailed description).
        """
        ...
    def render_rect_filled(self, output, x1, y1, x2, y2):  # -> None:
        """
        Draw a filled rectangle from (*x1*, *y1*) to (*x2*, *y2*).
        """
        ...
    def get_xheight(self, font, fontsize, dpi):
        """
        Get the xheight for the given *font* and *fontsize*.
        """
        ...
    def get_underline_thickness(self, font, fontsize, dpi):
        """
        Get the line thickness that matches the given font.  Used as a
        base unit for drawing lines such as in a fraction or radical.
        """
        ...
    def get_used_characters(self):
        """
        Get the set of characters that were used in the math
        expression.  Used by backends that need to subset fonts so
        they know which glyphs to include.
        """
        ...
    def get_sized_alternatives_for_symbol(
        self, fontname, sym
    ):  # -> list[tuple[Unknown, Unknown]]:
        """
        Override if your font provides multiple sizes of the same
        symbol.  Should return a list of symbols matching *sym* in
        various sizes.  The expression renderer will select the most
        appropriate size for a given situation from this list.
        """
        ...

class TruetypeFonts(Fonts):
    """
    A generic base class for all font setups that use Truetype fonts
    (through FT2Font).
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def get_xheight(self, fontname, fontsize, dpi): ...
    def get_underline_thickness(self, font, fontsize, dpi): ...
    def get_kern(
        self,
        font1,
        fontclass1,
        sym1,
        fontsize1,
        font2,
        fontclass2,
        sym2,
        fontsize2,
        dpi,
    ):  # -> Any | float:
        ...

class BakomaFonts(TruetypeFonts):
    """
    Use the Bakoma TrueType fonts for rendering.

    Symbols are strewn about a number of font files, each of which has
    its own proprietary 8-bit encoding.
    """

    _fontmap = ...
    def __init__(self, *args, **kwargs) -> None: ...

    _slanted_symbols = ...
    _size_alternatives = ...
    def get_sized_alternatives_for_symbol(
        self, fontname, sym
    ):  # -> list[tuple[Literal['rm'], Literal['(']] | tuple[Literal['ex'], Literal['¡']] | tuple[Literal['ex'], Literal['³']] | tuple[Literal['ex'], Literal['µ']] | tuple[Literal['ex'], Literal['Ã']]] | list[tuple[Literal['rm'], Literal[')']] | tuple[Literal['ex'], Literal['¢']] | tuple[Literal['ex'], Literal['´']] | tuple[Literal['ex'], Literal['¶']] | tuple[Literal['ex'], Literal['!']]] | list[tuple[Literal['cal'], Literal['{']] | tuple[Literal['ex'], Literal['©']] | tuple[Literal['ex'], Literal['n']] | tuple[Literal['ex'], Literal['½']] | tuple[Literal['ex'], Literal['(']]] | list[tuple[Literal['cal'], Literal['}']] | tuple[Literal['ex'], Literal['ª']] | tuple[Literal['ex'], Literal['o']] | tuple[Literal['ex'], Literal['¾']] | tuple[Literal['ex'], Literal[')']]] | list[tuple[Literal['rm'], Literal['[']] | tuple[Literal['ex'], Literal['£']] | tuple[Literal['ex'], Literal['h']] | tuple[Literal['ex'], Literal['"']]] | list[tuple[Literal['rm'], Literal[']']] | tuple[Literal['ex'], Literal['¤']] | tuple[Literal['ex'], Literal['i']] | tuple[Literal['ex'], Literal['#']]] | list[tuple[Literal['ex'], Literal['¥']] | tuple[Literal['ex'], Literal['j']] | tuple[Literal['ex'], Literal['¹']] | tuple[Literal['ex'], Literal['$']]] | list[tuple[Literal['ex'], Literal['¦']] | tuple[Literal['ex'], Literal['k']] | tuple[Literal['ex'], Literal['º']] | tuple[Literal['ex'], Literal['%']]] | list[tuple[Literal['ex'], Literal['§']] | tuple[Literal['ex'], Literal['l']] | tuple[Literal['ex'], Literal['»']] | tuple[Literal['ex'], Literal['&']]] | list[tuple[Literal['ex'], Literal['¨']] | tuple[Literal['ex'], Literal['m']] | tuple[Literal['ex'], Literal['¼']] | tuple[Literal['ex'], Literal['\'']]] | list[tuple[Literal['ex'], Literal['­']] | tuple[Literal['ex'], Literal['D']] | tuple[Literal['ex'], Literal['¿']] | tuple[Literal['ex'], Literal['*']]] | list[tuple[Literal['ex'], Literal['®']] | tuple[Literal['ex'], Literal['E']] | tuple[Literal['ex'], Literal['À']] | tuple[Literal['ex'], Literal['+']]] | list[tuple[Literal['ex'], Literal['p']] | tuple[Literal['ex'], Literal['q']] | tuple[Literal['ex'], Literal['r']] | tuple[Literal['ex'], Literal['s']]] | list[tuple[Literal['ex'], Literal['²']] | tuple[Literal['ex'], Literal['/']] | tuple[Literal['ex'], Literal['Â']] | tuple[Literal['ex'], Literal['-']]] | list[tuple[Literal['rm'], Literal['/']] | tuple[Literal['ex'], Literal['±']] | tuple[Literal['ex'], Literal['.']] | tuple[Literal['ex'], Literal['Ë']] | tuple[Literal['ex'], Literal[',']]] | list[tuple[Literal['rm'], Literal['^']] | tuple[Literal['ex'], Literal['b']] | tuple[Literal['ex'], Literal['c']] | tuple[Literal['ex'], Literal['d']]] | list[tuple[Literal['rm'], Literal['~']] | tuple[Literal['ex'], Literal['e']] | tuple[Literal['ex'], Literal['f']] | tuple[Literal['ex'], Literal['g']]] | list[tuple[Literal['cal'], Literal['h']] | tuple[Literal['ex'], Literal['D']]] | list[tuple[Literal['cal'], Literal['i']] | tuple[Literal['ex'], Literal['E']]]:
        ...

class UnicodeFonts(TruetypeFonts):
    """
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I'm aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol can not be found in the font.
    """

    _cmr10_substitutions = ...
    def __init__(self, *args, **kwargs) -> None: ...

    _slanted_symbols = ...
    def get_sized_alternatives_for_symbol(
        self, fontname, sym
    ):  # -> list[tuple[Unknown, str | Unknown]] | list[tuple[int, str]] | list[tuple[Literal['rm'], Literal['(']] | tuple[Literal['ex'], Literal['¡']] | tuple[Literal['ex'], Literal['³']] | tuple[Literal['ex'], Literal['µ']] | tuple[Literal['ex'], Literal['Ã']]] | list[tuple[Literal['rm'], Literal[')']] | tuple[Literal['ex'], Literal['¢']] | tuple[Literal['ex'], Literal['´']] | tuple[Literal['ex'], Literal['¶']] | tuple[Literal['ex'], Literal['!']]] | list[tuple[Literal['cal'], Literal['{']] | tuple[Literal['ex'], Literal['©']] | tuple[Literal['ex'], Literal['n']] | tuple[Literal['ex'], Literal['½']] | tuple[Literal['ex'], Literal['(']]] | list[tuple[Literal['cal'], Literal['}']] | tuple[Literal['ex'], Literal['ª']] | tuple[Literal['ex'], Literal['o']] | tuple[Literal['ex'], Literal['¾']] | tuple[Literal['ex'], Literal[')']]] | list[tuple[Literal['rm'], Literal['[']] | tuple[Literal['ex'], Literal['£']] | tuple[Literal['ex'], Literal['h']] | tuple[Literal['ex'], Literal['"']]] | list[tuple[Literal['rm'], Literal[']']] | tuple[Literal['ex'], Literal['¤']] | tuple[Literal['ex'], Literal['i']] | tuple[Literal['ex'], Literal['#']]] | list[tuple[Literal['ex'], Literal['¥']] | tuple[Literal['ex'], Literal['j']] | tuple[Literal['ex'], Literal['¹']] | tuple[Literal['ex'], Literal['$']]] | list[tuple[Literal['ex'], Literal['¦']] | tuple[Literal['ex'], Literal['k']] | tuple[Literal['ex'], Literal['º']] | tuple[Literal['ex'], Literal['%']]] | list[tuple[Literal['ex'], Literal['§']] | tuple[Literal['ex'], Literal['l']] | tuple[Literal['ex'], Literal['»']] | tuple[Literal['ex'], Literal['&']]] | list[tuple[Literal['ex'], Literal['¨']] | tuple[Literal['ex'], Literal['m']] | tuple[Literal['ex'], Literal['¼']] | tuple[Literal['ex'], Literal['\'']]] | list[tuple[Literal['ex'], Literal['­']] | tuple[Literal['ex'], Literal['D']] | tuple[Literal['ex'], Literal['¿']] | tuple[Literal['ex'], Literal['*']]] | list[tuple[Literal['ex'], Literal['®']] | tuple[Literal['ex'], Literal['E']] | tuple[Literal['ex'], Literal['À']] | tuple[Literal['ex'], Literal['+']]] | list[tuple[Literal['ex'], Literal['p']] | tuple[Literal['ex'], Literal['q']] | tuple[Literal['ex'], Literal['r']] | tuple[Literal['ex'], Literal['s']]] | list[tuple[Literal['ex'], Literal['²']] | tuple[Literal['ex'], Literal['/']] | tuple[Literal['ex'], Literal['Â']] | tuple[Literal['ex'], Literal['-']]] | list[tuple[Literal['rm'], Literal['/']] | tuple[Literal['ex'], Literal['±']] | tuple[Literal['ex'], Literal['.']] | tuple[Literal['ex'], Literal['Ë']] | tuple[Literal['ex'], Literal[',']]] | list[tuple[Literal['rm'], Literal['^']] | tuple[Literal['ex'], Literal['b']] | tuple[Literal['ex'], Literal['c']] | tuple[Literal['ex'], Literal['d']]] | list[tuple[Literal['rm'], Literal['~']] | tuple[Literal['ex'], Literal['e']] | tuple[Literal['ex'], Literal['f']] | tuple[Literal['ex'], Literal['g']]] | list[tuple[Literal['cal'], Literal['h']] | tuple[Literal['ex'], Literal['D']]] | list[tuple[Literal['cal'], Literal['i']] | tuple[Literal['ex'], Literal['E']]] | list[tuple[Unknown, Unknown]]:
        ...

class DejaVuFonts(UnicodeFonts):
    def __init__(self, *args, **kwargs) -> None: ...

class DejaVuSerifFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Serif fonts

    If a glyph is not found it will fallback to Stix Serif
    """

    _fontmap = ...

class DejaVuSansFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Sans fonts

    If a glyph is not found it will fallback to Stix Sans
    """

    _fontmap = ...

class StixFonts(UnicodeFonts):
    """
    A font handling class for the STIX fonts.

    In addition to what UnicodeFonts provides, this class:

    - supports "virtual fonts" which are complete alpha numeric
      character sets with different font styles at special Unicode
      code points, such as "Blackboard".

    - handles sized alternative characters for the STIXSizeX fonts.
    """

    _fontmap = ...
    _fallback_font = ...
    _sans = ...
    def __init__(self, *args, **kwargs) -> None: ...
    @functools.lru_cache()
    def get_sized_alternatives_for_symbol(
        self, fontname, sym
    ):  # -> list[tuple[Unknown, str | Unknown]] | list[tuple[int, str]]:
        ...

class StixSansFonts(StixFonts):
    """
    A font handling class for the STIX fonts (that uses sans-serif
    characters by default).
    """

    _sans = ...

SHRINK_FACTOR = ...
NUM_SIZE_LEVELS = ...

class FontConstantsBase:
    """
    A set of constants that controls how certain things, such as sub-
    and superscripts are laid out.  These are all metrics that can't
    be reliably retrieved from the font metrics in the font itself.
    """

    script_space = ...
    subdrop = ...
    sup1 = ...
    sub1 = ...
    sub2 = ...
    delta = ...
    delta_slanted = ...
    delta_integral = ...

class ComputerModernFontConstants(FontConstantsBase):
    script_space = ...
    subdrop = ...
    sup1 = ...
    sub1 = ...
    sub2 = ...
    delta = ...
    delta_slanted = ...
    delta_integral = ...

class STIXFontConstants(FontConstantsBase):
    script_space = ...
    sup1 = ...
    sub2 = ...
    delta = ...
    delta_slanted = ...
    delta_integral = ...

class STIXSansFontConstants(FontConstantsBase):
    script_space = ...
    sup1 = ...
    delta_slanted = ...
    delta_integral = ...

class DejaVuSerifFontConstants(FontConstantsBase): ...
class DejaVuSansFontConstants(FontConstantsBase): ...

_font_constant_mapping = ...

class Node:
    """A node in the TeX box model."""

    def __init__(self) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def get_kerning(self, next):  # -> float:
        ...
    def shrink(self):  # -> None:
        """
        Shrinks one level smaller.  There are only three levels of
        sizes, after which things will no longer get smaller.
        """
        ...
    def render(self, output, x, y):  # -> None:
        """Render this node."""
        ...

class Box(Node):
    """A node with a physical location."""

    def __init__(self, width, height, depth) -> None: ...
    def shrink(self):  # -> None:
        ...
    def render(self, output, x1, y1, x2, y2):  # -> None:
        ...

class Vbox(Box):
    """A box with only height (zero width)."""

    def __init__(self, height, depth) -> None: ...

class Hbox(Box):
    """A box with only width (zero height and depth)."""

    def __init__(self, width) -> None: ...

class Char(Node):
    """
    A single character.

    Unlike TeX, the font information and metrics are stored with each `Char`
    to make it easier to lookup the font metrics when needed.  Note that TeX
    boxes have a width, height, and depth, unlike Type1 and TrueType which use
    a full bounding box and an advance in the x-direction.  The metrics must
    be converted to the TeX model, and the advance (if different from width)
    must be converted into a `Kern` node when the `Char` is added to its parent
    `Hlist`.
    """

    def __init__(self, c, state) -> None: ...
    def __repr__(self):  # -> LiteralString:
        ...
    def is_slanted(self): ...
    def get_kerning(self, next):
        """
        Return the amount of kerning between this and the given character.

        This method is called when characters are strung together into `Hlist`
        to create `Kern` nodes.
        """
        ...
    def render(self, output, x, y):  # -> None:
        ...
    def shrink(self):  # -> None:
        ...

class Accent(Char):
    """
    The font metrics need to be dealt with differently for accents,
    since they are already offset correctly from the baseline in
    TrueType fonts.
    """

    def shrink(self):  # -> None:
        ...
    def render(self, output, x, y):  # -> None:
        ...

class List(Box):
    """A list of nodes (either horizontal or vertical)."""

    def __init__(self, elements) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def shrink(self):  # -> None:
        ...

class Hlist(List):
    """A horizontal list of boxes."""

    def __init__(self, elements, w=..., m=..., do_kern=...) -> None: ...
    def kern(self):  # -> None:
        """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
        ...
    def hpack(self, w=..., m=...):  # -> None:
        r"""
        Compute the dimensions of the resulting boxes, and adjust the glue if
        one of those dimensions is pre-specified.  The computed sizes normally
        enclose all of the material inside the new box; but some items may
        stick out if negative glue is used, if the box is overfull, or if a
        ``\vbox`` includes other boxes that have been shifted left.

        Parameters
        ----------
        w : float, default: 0
            A width.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose width is 'exactly' *w*; or a box
            with the natural width of the contents, plus *w* ('additional').

        Notes
        -----
        The defaults produce a box with the natural width of the contents.
        """
        ...

class Vlist(List):
    """A vertical list of boxes."""

    def __init__(self, elements, h=..., m=...) -> None: ...
    def vpack(self, h=..., m=..., l=...):  # -> None:
        """
        Compute the dimensions of the resulting boxes, and to adjust the glue
        if one of those dimensions is pre-specified.

        Parameters
        ----------
        h : float, default: 0
            A height.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose height is 'exactly' *h*; or a box
            with the natural height of the contents, plus *h* ('additional').
        l : float, default: np.inf
            The maximum height.

        Notes
        -----
        The defaults produce a box with the natural height of the contents.
        """
        ...

class Rule(Box):
    """
    A solid black rectangle.

    It has *width*, *depth*, and *height* fields just as in an `Hlist`.
    However, if any of these dimensions is inf, the actual value will be
    determined by running the rule up to the boundary of the innermost
    enclosing box.  This is called a "running dimension".  The width is never
    running in an `Hlist`; the height and depth are never running in a `Vlist`.
    """

    def __init__(self, width, height, depth, state) -> None: ...
    def render(self, output, x, y, w, h):  # -> None:
        ...

class Hrule(Rule):
    """Convenience class to create a horizontal rule."""

    def __init__(self, state, thickness=...) -> None: ...

class Vrule(Rule):
    """Convenience class to create a vertical rule."""

    def __init__(self, state) -> None: ...

_GlueSpec = ...

class Glue(Node):
    """
    Most of the information in this object is stored in the underlying
    ``_GlueSpec`` class, which is shared between multiple glue objects.
    (This is a memory optimization which probably doesn't matter anymore, but
    it's easier to stick to what TeX does.)
    """

    def __init__(self, glue_type) -> None: ...
    def shrink(self):  # -> None:
        ...

class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements) -> None: ...

class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements) -> None: ...

class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """

    height = ...
    depth = ...
    def __init__(self, width) -> None: ...
    def __repr__(self):  # -> LiteralString:
        ...
    def shrink(self):  # -> None:
        ...

class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, height, depth, state, always=..., factor=...) -> None: ...

class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c, width, state, always=..., char_class=...) -> None: ...

def ship(box, xy=...):  # -> Output:
    """
    Ship out *box* at offset *xy*, converting it to an `Output`.

    Since boxes can be inside of boxes inside of boxes, the main work of `ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become local variables here.
    """
    ...

def Error(msg):  # -> ParserElement:
    """Helper class to raise parser errors."""
    ...

class ParserState:
    """
    Parser state.

    States are pushed and popped from a stack as necessary, and the "current"
    state is always at the top of the stack.

    Upon entering and leaving a group { } or math/non-math, the stack is pushed
    and popped accordingly.
    """

    def __init__(self, fontset, font, font_class, fontsize, dpi) -> None: ...
    def copy(self):  # -> Self@ParserState:
        ...
    @property
    def font(self):  # -> Unknown:
        ...
    @font.setter
    def font(self, name):  # -> None:
        ...
    def get_current_underline_thickness(self):
        """Return the underline thickness for this state."""
        ...

def cmd(expr, args):
    r"""
    Helper to define TeX commands.

    ``cmd("\cmd", args)`` is equivalent to
    ``"\cmd" - (args | Error("Expected \cmd{arg}{...}"))`` where the names in
    the error message are taken from element names in *args*.  If *expr*
    already includes arguments (e.g. "\cmd{arg}{...}"), then they are stripped
    when constructing the parse element, but kept (and *expr* is used as is) in
    the error message.
    """
    ...

class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    class _MathStyle(enum.Enum):
        DISPLAYSTYLE = ...
        TEXTSTYLE = ...
        SCRIPTSTYLE = ...
        SCRIPTSCRIPTSTYLE = ...
    _binary_operators = ...
    _relation_symbols = ...
    _arrow_symbols = ...
    _spaced_symbols = ...
    _punctuation_symbols = ...
    _overunder_symbols = ...
    _overunder_functions = ...
    _dropsub_symbols = ...
    _fontnames = ...
    _function_names = ...
    _ambi_delims = ...
    _left_delims = ...
    _right_delims = ...
    _delims = ...
    def __init__(self) -> None: ...
    def parse(self, s, fonts_object, fontsize, dpi):  # -> ParseResults:
        """
        Parse expression *s* using the given *fonts_object* for
        output, at the given *fontsize* and *dpi*.

        Returns the parse tree of `Node` instances.
        """
        ...
    def get_state(self):  # -> ParserState:
        """Get the current `State` of the parser."""
        ...
    def pop_state(self):  # -> None:
        """Pop a `State` off of the stack."""
        ...
    def push_state(self):  # -> None:
        """Push a new `State` onto the stack, copying the current state."""
        ...
    def main(self, s, loc, toks):  # -> list[Hlist]:
        ...
    def math_string(self, s, loc, toks):  # -> ParseResults:
        ...
    def math(self, s, loc, toks):  # -> list[Hlist]:
        ...
    def non_math(self, s, loc, toks):  # -> list[Hlist]:
        ...
    float_literal = ...
    _space_widths = ...
    def space(self, s, loc, toks):  # -> list[Kern]:
        ...
    def customspace(self, s, loc, toks):  # -> list[Kern]:
        ...
    def symbol(self, s, loc, toks):  # -> list[Char] | list[Hlist]:
        ...
    def unknown_symbol(self, s, loc, toks): ...

    _accent_map = ...
    _wide_accents = ...
    def accent(self, s, loc, toks):  # -> Vlist:
        ...
    def function(self, s, loc, toks):  # -> Hlist:
        ...
    def operatorname(self, s, loc, toks):  # -> Hlist:
        ...
    def start_group(self, s, loc, toks):  # -> list[Unknown]:
        ...
    def group(self, s, loc, toks):  # -> list[Hlist]:
        ...
    def required_group(self, s, loc, toks):  # -> Hlist:
        ...
    optional_group = ...
    def end_group(self, s, loc, toks):  # -> list[Unknown]:
        ...
    def font(self, s, loc, toks):  # -> list[Unknown]:
        ...
    def is_overunder(self, nucleus):  # -> bool:
        ...
    def is_dropsub(self, nucleus):  # -> bool:
        ...
    def is_slanted(self, nucleus):  # -> Literal[False]:
        ...
    def is_between_brackets(self, s, loc):  # -> Literal[False]:
        ...
    def subsuper(self, s, loc, toks): ...
    def style_literal(self, s, loc, toks):  # -> _MathStyle:
        ...
    def genfrac(self, s, loc, toks):  # -> Hlist | list[Hlist]:
        ...
    def frac(self, s, loc, toks):  # -> Hlist | list[Hlist]:
        ...
    def dfrac(self, s, loc, toks):  # -> Hlist | list[Hlist]:
        ...
    def binom(self, s, loc, toks):  # -> Hlist | list[Hlist]:
        ...
    underset = ...
    def sqrt(self, s, loc, toks):  # -> list[Hlist]:
        ...
    def overline(self, s, loc, toks):  # -> list[Hlist]:
        ...
    def auto_delim(self, s, loc, toks):  # -> Hlist:
        ...
