"""Bokeh dashboard showing real-time metrics"""

from datetime import datetime, timedelta
import numbers
import logging
import functools
from collections import deque
from weakref import WeakSet
from typing import MutableMapping, MutableSet, List, Callable   # noqa: F401

from aiokatcp import Sensor, Reading

from bokeh.document import Document
from bokeh.application.handlers.handler import Handler
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.application.application import Application


logger = logging.getLogger(__name__)


def _convert_timestamp(posix_timestamp: float) -> datetime:
    return datetime.utcfromtimestamp(posix_timestamp)


class Watcher:
    """Observe and collect data for a single sensor

    Refer to :class:`Dashboard` for the meaning of `window` and `rollover`.
    """
    def __init__(self, dashboard: 'Dashboard', sensor: Sensor,
                 window: float, rollover: int) -> None:
        self.dashboard = dashboard
        self.sensor = sensor
        self.window = window
        self.rollover = rollover
        # TODO: use typing.Deque in type hint after migration to Python 3.6
        self._readings = deque()        # type: deque
        self.sensor.attach(self._update)
        self._update(self.sensor, self.sensor.reading)

    def close(self) -> None:
        self.sensor.detach(self._update)

    def make_data_source(self) -> ColumnDataSource:
        data = {
            'time': [_convert_timestamp(reading.timestamp) for reading in self._readings],
            'value': [reading.value for reading in self._readings]
        }
        return ColumnDataSource(data, name='data_source ' + self.sensor.name)

    def _update(self, sensor: Sensor, reading: Reading[float]) -> None:
        self._readings.append(reading)
        if (self._readings[-1].timestamp - self._readings[0].timestamp > self.window
                or len(self._readings) > self.rollover):
            self._readings.popleft()
        update = {
            'time': [_convert_timestamp(reading.timestamp)],
            'value': [reading.value]
        }
        name = 'data_source ' + sensor.name

        def doc_update(doc):
            data_source = doc.get_model_by_name(name)
            data_source.stream(update, rollover=len(self._readings))

        self.dashboard.update_documents(doc_update)


class Dashboard(Handler):
    """Bokeh dashboard showing sensor values.

    Sensor values are recorded and displayed through graphs. To keep the
    graph size down (more to avoid overloading the browser/network than for
    memory constraints), old values are discarded once either they are
    older than `window` or there are more than `rollover` samples.

    Parameters
    ----------
    sensors
        Sensors to display. Non-numeric sensors are ignored.
    window
        Maximum length of time (in seconds) to keep samples.
    rollover
        Maximum number of samples to keep (per sensor).
    """
    def __init__(self, sensors: MutableMapping[str, Sensor],
                 window: float = 1200.0, rollover: int = 10000) -> None:
        super().__init__()
        self._watchers = []                # type: List[Watcher]
        self._docs = WeakSet()             # type: MutableSet[Document]
        for sensor in sensors.values():
            if issubclass(sensor.stype, numbers.Real):
                self._watchers.append(Watcher(self, sensor, window, rollover))

    def modify_document(self, doc: Document) -> None:
        plots = []
        renderers = []         # type: List
        for watcher in self._watchers:
            data_source = watcher.make_data_source()
            plot = figure(title=watcher.sensor.name, plot_width=350, plot_height=350,
                          x_axis_label='time', x_axis_type='datetime', y_axis_label='value')
            plot.step('time', 'value', source=data_source, mode='after')
            plots.append(plot)
            renderers.extend(plot.x_range.renderers)
        # Create a single data range so that all plots show the same time window
        data_range = DataRange1d()
        data_range.renderers = renderers
        data_range.follow = 'end'
        data_range.default_span = timedelta(seconds=1)
        data_range.follow_interval = timedelta(seconds=120)
        for plot in plots:
            plot.x_range = data_range

        doc.add_root(gridplot(plots, ncols=3))
        logger.debug('Created document with %d plots', len(plots))
        self._docs.add(doc)

    def on_server_unloaded(self, server_context) -> None:
        for watcher in self._watchers:
            watcher.close()
        self._watchers.clear()

    def update_documents(self, callback: Callable[[Document], None]) -> None:
        for doc in self._docs:
            doc.add_next_tick_callback(functools.partial(callback, doc))


def start_dashboard(dashboard: Dashboard, port: int) -> None:
    app = Application()
    app.add(dashboard)
    server = Server(app, port=port)
    server.start()
