"""Bokeh dashboard showing real-time metrics"""

from datetime import datetime, timedelta
import logging
import functools
from collections import deque
from weakref import WeakSet
from typing import Mapping, MutableSet, List, Callable, Iterable   # noqa: F401

import numpy as np

from aiokatcp import Sensor, Reading

from bokeh.document import Document
from bokeh.application.handlers.handler import Handler
from bokeh.models import ColumnDataSource, DataRange1d
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.palettes import Category10
from bokeh.server.server import Server
from bokeh.application.application import Application


logger = logging.getLogger(__name__)
PALETTE = Category10[10]


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

    def _update(self, sensor: Sensor, reading: Reading[float]) -> None:
        self._readings.append(reading)
        if (self._readings[-1].timestamp - self._readings[0].timestamp > self.window
                or len(self._readings) > self.rollover):
            self._readings.popleft()


class LineWatcher(Watcher):
    """Watcher for drawing line graphs"""
    def make_data_source(self) -> ColumnDataSource:
        data = {
            'time': [_convert_timestamp(reading.timestamp) for reading in self._readings],
            'value': [reading.value for reading in self._readings]
        }
        return ColumnDataSource(data, name='data_source ' + self.sensor.name)

    def _update(self, sensor: Sensor, reading: Reading[float]) -> None:
        super()._update(sensor, reading)
        update = {
            'time': [_convert_timestamp(reading.timestamp)],
            'value': [reading.value]
        }
        name = 'data_source ' + sensor.name

        def doc_update(doc):
            data_source = doc.get_model_by_name(name)
            data_source.stream(update, rollover=len(self._readings))

        self.dashboard.update_documents(doc_update)


class HistogramWatcher(Watcher):
    def make_data_source(self) -> ColumnDataSource:
        return ColumnDataSource(self._data, name='data_source ' + self.sensor.name)

    def _update(self, sensor: Sensor, reading: Reading[float]) -> None:
        super()._update(sensor, reading)
        values = [reading.value for reading in self._readings]
        # Based on https://bokeh.pydata.org/en/latest/docs/gallery/histogram.html
        hist, edges = np.histogram(values, bins='auto')
        self._data = {
            'top': hist,
            'bottom': [0] * len(hist),
            'left': edges[:-1],
            'right': edges[1:]
        }
        name = 'data_source ' + sensor.name

        def doc_update(doc):
            data_source = doc.get_model_by_name(name)
            data_source.data = self._data

        self.dashboard.update_documents(doc_update)


class Dashboard(Handler):
    """Bokeh dashboard showing sensor values.

    Sensor values are recorded and displayed through graphs. To keep the
    graph size down (more to avoid overloading the browser/network than for
    memory constraints), old values are discarded once either they are
    older than `window` or there are more than `rollover` samples.

    Parameters
    ----------
    line_sensors
        Sensors to display as line graphs. Each element is a list of sensors
        to plot on a single graph.
    histogram_sensors
        Sensors to display as histograms. Each sensor update contributes one
        entry on the histogram.
    window
        Maximum length of time (in seconds) to keep samples.
    rollover
        Maximum number of samples to keep (per sensor).
    """
    def __init__(self,
                 line_sensors: Iterable[Iterable[Sensor]],
                 histogram_sensors: Iterable[Sensor],
                 window: float = 1200.0, rollover: int = 10000) -> None:
        super().__init__()
        self._line_watchers = []           # type: List[List[LineWatcher]]
        self._histogram_watchers = []      # type: List[HistogramWatcher]
        self._docs = WeakSet()             # type: MutableSet[Document]
        for sensors in line_sensors:
            watchers = [LineWatcher(self, sensor, window, rollover) for sensor in sensors]
            self._line_watchers.append(watchers)
        for sensor in histogram_sensors:
            watcher = HistogramWatcher(self, sensor, window, rollover)
            self._histogram_watchers.append(watcher)

    def modify_document(self, doc: Document) -> None:
        plots = []
        line_renderers = []         # type: List
        for watchers in self._line_watchers:
            plot = figure(plot_width=350, plot_height=350,
                          x_axis_label='time', x_axis_type='datetime', y_axis_label='value')
            for i, watcher in enumerate(watchers):
                data_source = watcher.make_data_source()
                plot.step('time', 'value', source=data_source, mode='after',
                          legend=watcher.sensor.name,
                          color=PALETTE[i])
            plot.legend.location = 'top_left'
            plots.append(plot)
            line_renderers.extend(plot.x_range.renderers)
        # Create a single data range so that all line plots show the same time window
        data_range = DataRange1d()
        data_range.renderers = line_renderers
        data_range.follow = 'end'
        data_range.default_span = timedelta(seconds=1)
        data_range.follow_interval = timedelta(seconds=120)
        for plot in plots:
            plot.x_range = data_range

        for watcher in self._histogram_watchers:
            plot = figure(plot_width=350, plot_height=350,
                          x_axis_label='value', y_axis_label='frequency')
            data_source = watcher.make_data_source()
            plot.quad(top='top', bottom='bottom', left='left', right='right',
                      source=data_source)
            plots.append(plot)

        doc.add_root(gridplot(plots, ncols=3))
        logger.debug('Created document with %d plots', len(plots))
        self._docs.add(doc)

    def on_server_unloaded(self, server_context) -> None:
        for watchers in self._line_watchers:
            for watcher in watchers:
                watcher.close()
        for watcher in self._histogram_watchers:
            watcher.close()
        self._line_watchers.clear()
        self._histogram_watchers.clear()

    def update_documents(self, callback: Callable[[Document], None]) -> None:
        for doc in self._docs:
            doc.add_next_tick_callback(functools.partial(callback, doc))


def make_dashboard(sensors: Mapping[str, Sensor]) -> Dashboard:
    """Build a dashboard using a standard set of sensors"""
    line_sensors = [
        [sensors['active-chunks']],
        [sensors['output-seconds-total']],
        [sensors['output-chunks-total']],
        [sensors['input-bytes-total'], sensors['output-bytes-total']],
        [sensors['input-heaps-total']],
        [sensors['input-incomplete-heaps-total']]
    ]
    histogram_sensors = [sensors['output-seconds']]
    return Dashboard(line_sensors, histogram_sensors)


def start_dashboard(dashboard: Dashboard, port: int) -> None:
    app = Application()
    app.add(dashboard)
    server = Server(app, port=port)
    server.start()
