"""
This type stub file was generated by pyright.
"""

from matplotlib import ticker, units

"""
Plotting of string "category" data: ``plot(['d', 'f', 'a'], [1, 2, 3])`` will
plot three points with x-axis values of 'd', 'f', 'a'.

See :doc:`/gallery/lines_bars_and_markers/categorical_variables` for an
example.

The module uses Matplotlib's `matplotlib.units` mechanism to convert from
strings to integers and provides a tick locator, a tick formatter, and the
`.UnitData` class that creates and stores the string-to-integer mapping.
"""
_log = ...

class StrCategoryConverter(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):  # -> Any:
        """
        Convert strings in *value* to floats using mapping information stored
        in the *unit* object.

        Parameters
        ----------
        value : str or iterable
            Value or list of values to be converted.
        unit : `.UnitData`
            An object mapping strings to integers.
        axis : `~matplotlib.axis.Axis`
            The axis on which the converted value is plotted.

            .. note:: *axis* is unused.

        Returns
        -------
        float or `~numpy.ndarray` of float
        """
        ...
    @staticmethod
    def axisinfo(unit, axis):  # -> AxisInfo:
        """
        Set the default axis ticks and labels.

        Parameters
        ----------
        unit : `.UnitData`
            object string unit information for value
        axis : `~matplotlib.axis.Axis`
            axis for which information is being set

            .. note:: *axis* is not used

        Returns
        -------
        `~matplotlib.units.AxisInfo`
            Information to support default tick labeling

        """
        ...
    @staticmethod
    def default_units(data, axis):
        """
        Set and update the `~matplotlib.axis.Axis` units.

        Parameters
        ----------
        data : str or iterable of str
        axis : `~matplotlib.axis.Axis`
            axis on which the data is plotted

        Returns
        -------
        `.UnitData`
            object storing string to integer mapping
        """
        ...

class StrCategoryLocator(ticker.Locator):
    """Tick at every integer mapping of the string data."""

    def __init__(self, units_mapping) -> None:
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        ...
    def __call__(self):  # -> list[Unknown]:
        ...
    def tick_values(self, vmin, vmax):  # -> list[Unknown]:
        ...

class StrCategoryFormatter(ticker.Formatter):
    """String representation of the data at every tick."""

    def __init__(self, units_mapping) -> None:
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        ...
    def __call__(self, x, pos=...):  # -> str:
        ...
    def format_ticks(self, values):  # -> list[str]:
        ...

class UnitData:
    def __init__(self, data=...) -> None:
        """
        Create mapping between unique categorical values and integer ids.

        Parameters
        ----------
        data : iterable
            sequence of string values
        """
        ...
    def update(self, data):  # -> None:
        """
        Map new values to integer identifiers.

        Parameters
        ----------
        data : iterable of str or bytes

        Raises
        ------
        TypeError
            If elements in *data* are neither str nor bytes.
        """
        ...
