#!/usr/bin/env python3
"""Capture L1 flags from the SPEAD stream(s) produced by cal.

We adopt a naive strategy and store the flags for each heap in a single
object. These objects will be later picked up by the trawler process
and inserted into the appropriate bucket in S3 from where they will be
picked up by katdal.

"""

import logging
import signal
import asyncio

import aiomonitor
import katsdptelstate
import katsdpservices

from katsdpdatawriter.flag_writer import FlagWriterServer
from katsdpdatawriter.spead_write import add_common_args, chunk_store_from_args
from katsdpdatawriter.dashboard import make_dashboard, start_dashboard


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
    add_common_args(parser)
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
    parser.set_defaults(telstate='localhost', port=2052)

    args = parser.parse_args()
    if args.telstate is None:
        parser.error('--telstate is required')
    if args.flags_ibv and args.flags_interface is None:
        parser.error("--flags-ibv requires --flags-interface")
    if args.rename_src and args.new_name is None:
        parser.error('--rename-src requires --new-name')

    chunk_store = chunk_store_from_args(parser, args)
    loop = asyncio.get_event_loop()
    server = FlagWriterServer(args.host, args.port, loop, args.flags_spead,
                              args.flags_interface, args.flags_ibv,
                              chunk_store, args.obj_size_mb * 1e6,
                              args.telstate,
                              args.flags_name,
                              args.new_name if args.new_name is not None else args.flags_name,
                              args.rename_src, args.s3_endpoint_url,
                              args.workers)
    if args.dashboard:
        dashboard = make_dashboard(server.sensors)
        start_dashboard(dashboard, args.dashboard_port)

    if args.aiomonitor:
        with aiomonitor.start_monitor(loop=loop,
                                      port=args.aiomonitor_port,
                                      console_port=args.aioconsole_port,
                                      locals=locals()):
            loop.run_until_complete(run(loop, server))
    else:
        loop.run_until_complete(run(loop, server))
    loop.close()
