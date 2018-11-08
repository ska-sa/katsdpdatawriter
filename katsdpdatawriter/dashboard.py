"""Bokeh dashboard showing real-time metrics"""

import datetime
import numbers
import logging
import functools
from collections import deque
from weakref import WeakSet
from typing import MutableMapping, Dict, List, Union, MutableSet   # noqa: F401

import tornado.ioloop

from aiokatcp import Sensor

from bokeh.document import Document
from bokeh.application.handlers.handler import Handler
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.application.application import Application


logger = logging.getLogger(__name__)
Value = Union[datetime.datetime, float]


class Dashboard(Handler):
    def __init__(self, sensors: MutableMapping[str, Sensor],
                 rollover: int = 10000, period: float = 0.1) -> None:
        super().__init__()
        self._sensors = []                # type: List[Sensor]
        # TODO: use typing.Deque in type hint after migration to Python 3.6
        self._cache = {'time': deque()}   # type: Dict[str, deque]
        self._docs = WeakSet()            # type: MutableSet[Document]
        self._rollover = rollover
        for sensor in sensors.values():
            if issubclass(sensor.stype, numbers.Real):
                self._sensors.append(sensor)
                self._cache[sensor.name] = deque()
        self._update()
        self._callback_handle = tornado.ioloop.PeriodicCallback(
            self._update, 1000 * period)
        self._callback_handle.start()

    def modify_document(self, doc: Document) -> None:
        plots = []
        data = {key: list(value) for key, value in self._cache.items()}
        data_source = ColumnDataSource(data, name='data_source')
        for sensor in self._sensors:
            plot = figure(title=sensor.name, plot_width=350, plot_height=350,
                          x_axis_label='time', x_axis_type='datetime', y_axis_label='value')
            plot.x_range.follow = 'end'
            plot.x_range.follow_interval = datetime.timedelta(seconds=120)
            plot.line('time', sensor.name, source=data_source)
            plots.append(plot)
        doc.add_root(gridplot(plots, ncols=3))
        logger.debug('Created document with %d plots', len(plots))
        self._docs.add(doc)

    def on_server_unloaded(self, server_context) -> None:
        self._callback_handle.stop()

    def _update_document(self, doc: Document, row: Dict[str, List[Value]]) -> None:
        data_source = doc.get_model_by_name('data_source')
        data_source.stream(row, rollover=self._rollover)

    def _update(self) -> None:
        """Sample all the sensors and update the document"""
        row = {'time': [datetime.datetime.utcnow()]}
        for sensor in self._sensors:
            row[sensor.name] = [sensor.value]
        for doc in self._docs:
            doc.add_next_tick_callback(functools.partial(self._update_document, doc, row))
        for key, values in row.items():
            self._cache[key].extend(values)
            while len(self._cache[key]) > self._rollover:
                self._cache[key].popleft()


def start_dashboard(dashboard: Dashboard, port: int) -> None:
    app = Application()
    app.add(dashboard)
    server = Server(app, port=port)
    server.start()
