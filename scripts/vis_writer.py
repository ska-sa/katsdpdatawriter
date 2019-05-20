#!/usr/bin/env python3

import asyncio
import signal
import logging

import katsdpservices
import katsdptelstate

from katsdpdatawriter.vis_writer import VisibilityWriterServer
from katsdpdatawriter.spead_write import add_common_args, chunk_store_from_args, ChunkParams
from katsdpdatawriter.dashboard import make_dashboard, start_dashboard


def on_shutdown(loop: asyncio.AbstractEventLoop, server: VisibilityWriterServer) -> None:
    # in case the exit code below borks, we allow shutdown via traditional means
    loop.remove_signal_handler(signal.SIGINT)
    loop.remove_signal_handler(signal.SIGTERM)
    server.halt()


async def run(loop: asyncio.AbstractEventLoop, server: VisibilityWriterServer) -> None:
    await server.start()
    for sig in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(sig, lambda: on_shutdown(loop, server))
    logger.info("Started visibility writer server.")
    await server.join()


if __name__ == '__main__':
    katsdpservices.setup_logging()
    logger = logging.getLogger('vis_writer')
    katsdpservices.setup_restart()

    parser = katsdpservices.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--l0-spead', default=':7200', metavar='ENDPOINTS',
                        type=katsdptelstate.endpoint.endpoint_list_parser(7200),
                        help='Source port/multicast groups for L0 SPEAD stream. '
                             '[default=%(default)s]')
    parser.add_argument('--l0-interface', metavar='INTERFACE',
                        help='Network interface to subscribe to for L0 stream. '
                             '[default=auto]')
    parser.add_argument('--l0-name', default='sdp_l0', metavar='NAME',
                        help='Name of L0 stream from ingest [default=%(default)s]')
    parser.add_argument('--l0-ibv', action='store_true',
                        help='Use ibverbs acceleration to receive L0 stream [default=no]')
    parser.set_defaults(telstate='localhost', port=2046)
    args = parser.parse_args()

    if args.l0_ibv and args.l0_interface is None:
        parser.error('--l0-ibv requires --l0-interface')
    if args.rename_src and args.new_name is None:
        parser.error('--rename-src requires --new-name')

    # Connect to object store
    chunk_store = chunk_store_from_args(parser, args)
    loop = asyncio.get_event_loop()
    server = VisibilityWriterServer(args.host, args.port, loop, args.l0_spead,
                                    args.l0_interface, args.l0_ibv,
                                    chunk_store, ChunkParams.from_args(args),
                                    args.telstate,
                                    args.l0_name,
                                    args.new_name if args.new_name is not None else args.l0_name,
                                    args.rename_src,
                                    args.s3_endpoint_url,
                                    args.workers, args.buffer_dumps)
    if args.dashboard_port is not None:
        dashboard = make_dashboard(server.sensors)
        start_dashboard(dashboard, args)

    with katsdpservices.start_aiomonitor(loop, args, locals()):
        loop.run_until_complete(run(loop, server))
    loop.close()
