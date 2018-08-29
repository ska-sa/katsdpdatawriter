#!/usr/bin/env python3

import asyncio
import signal
import logging

import aiomonitor
import katsdpservices
import katsdptelstate

from katsdpdatawriter.vis_writer import VisibilityWriterServer
from katsdpdatawriter.spead_write import add_chunk_store_args, chunk_store_from_args


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
    add_chunk_store_args(parser)
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
    parser.add_argument('--obj-size-mb', type=float, default=10., metavar='MB',
                        help='Target object size in MB [default=%(default)s]')
    parser.add_argument('--workers', type=int, default=50,
                        help='Threads to use for writing chunks')
    parser.add_argument('--no-aiomonitor', dest='aiomonitor', action='store_false',
                        help='Disable aiomonitor debugging server')
    parser.add_argument('--aiomonitor-port', type=int, default=aiomonitor.MONITOR_PORT,
                        help='port for aiomonitor [default=%(default)s]')
    parser.add_argument('--aioconsole-port', type=int, default=aiomonitor.CONSOLE_PORT,
                        help='port for aioconsole [default=%(default)s]')
    parser.add_argument('-p', '--port', type=int, default=2046, metavar='N',
                        help='KATCP host port [default=%(default)s]')
    parser.add_argument('-a', '--host', default="", metavar='HOST',
                        help='KATCP host address [default=all hosts]')
    parser.set_defaults(telstate='localhost')
    args = parser.parse_args()

    if args.l0_ibv and args.l0_interface is None:
        parser.error('--l0-ibv requires --l0-interface')

    # Connect to object store and save config in telstate
    chunk_store = chunk_store_from_args(parser, args)
    telstate_l0 = args.telstate.view(args.l0_name)
    if args.s3_endpoint_url:
        telstate_l0.add('s3_endpoint_url', args.s3_endpoint_url, immutable=True)

    loop = asyncio.get_event_loop()
    server = VisibilityWriterServer(args.host, args.port, loop, args.l0_spead,
                                    args.l0_interface, args.l0_ibv,
                                    chunk_store, args.obj_size_mb * 1e6,
                                    telstate_l0, args.l0_name, args.workers)

    if args.aiomonitor:
        with aiomonitor.start_monitor(loop=loop,
                                      port=args.aiomonitor_port,
                                      console_port=args.aioconsole_port,
                                      locals=locals()):
            loop.run_until_complete(run(loop, server))
    else:
        loop.run_until_complete(run(loop, server))
    loop.close()
