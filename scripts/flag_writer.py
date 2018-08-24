#!/usr/bin/env python3
"""Capture L1 flags from the SPEAD stream(s) produced by cal.

We adopt a naive strategy and store the flags for each heap in a single
object. These objects will be later picked up by the trawler process
and inserted into the appropriate bucket in S3 from where they will be
picked up by katdal.

"""

import os
import logging
import signal
import asyncio

import aiomonitor
import katsdptelstate
import katsdpservices
import katdal.chunkstore_npy

from katsdpdatawriter.flag_writer import FlagWriterServer


def on_shutdown(loop: asyncio.AbstractEventLoop, server: FlagWriterServer) -> None:
    # in case the exit code below borks, we allow shutdown via traditional means
    loop.remove_signal_handler(signal.SIGINT)
    loop.remove_signal_handler(signal.SIGTERM)
    server.halt()


async def run(loop: asyncio.AbstractEventLoop, server: FlagWriterServer) -> None:
    await server.start()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: on_shutdown(loop, server))
    logger.info("Started flag writer server.")
    await server.join()


if __name__ == '__main__':
    katsdpservices.setup_logging()
    logger = logging.getLogger("flag_writer")
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    parser.add_argument('--npy-path', default="/var/kat/data", metavar='NPYPATH',
                        help='Root in which to write flag dumps in npy format.')
    parser.add_argument('--flags-spead', default=':7202', metavar='ENDPOINTS',
                        type=katsdptelstate.endpoint.endpoint_list_parser(7202),
                        help='Source port/multicast groups for flags SPEAD streams. '
                             '[default=%(default)s]')
    parser.add_argument('--flags-interface', metavar='INTERFACE',
                        help='Network interface to subscribe to for flag streams. '
                             '[default=auto]')
    parser.add_argument('--flags-name', type=str, default='sdp_l1_flags',
                        help='name for the flags stream. [default=%(default)s]', metavar='NAME')
    parser.add_argument('--flags-ibv', action='store_true',
                        help='Use ibverbs acceleration to receive flags')
    parser.add_argument('--workers', type=int, default=50,
                        help='Threads to use for writing chunks')
    parser.add_argument('--no-aiomonitor', dest='aiomonitor', action='store_false',
                        help='Disable aiomonitor debugging server')
    parser.add_argument('--aiomonitor-port', type=int, default=aiomonitor.MONITOR_PORT,
                        help='port for aiomonitor [default=%(default)s]')
    parser.add_argument('--aioconsole-port', type=int, default=aiomonitor.CONSOLE_PORT,
                        help='port for aioconsole [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2052, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [default=all hosts]')

    args = parser.parse_args()
    if args.telstate is None:
        parser.error('--telstate is required')
    if args.flags_ibv and args.flags_interface is None:
        parser.error("--flags-ibv requires --flags-interface")
    if not os.path.isdir(args.npy_path):
        parser.error("Specified NPY path, %s, does not exist.", args.npy_path)

    loop = asyncio.get_event_loop()

    chunk_store = katdal.chunkstore_npy.NpyFileChunkStore(args.npy_path)
    server = FlagWriterServer(args.host, args.port, loop, args.flags_spead,
                              args.flags_interface, args.flags_ibv, chunk_store,
                              args.telstate, args.flags_name, args.workers)

    if args.aiomonitor:
        with aiomonitor.start_monitor(loop=loop,
                                      port=args.aiomonitor_port,
                                      console_port=args.aioconsole_port,
                                      locals=locals()):
            loop.run_until_complete(run(loop, server))
    else:
        loop.run_until_complete(run(loop, server))
    loop.close()
