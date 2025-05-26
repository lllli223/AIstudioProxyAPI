import asyncio
import multiprocessing

from stream import main
def start(queue: multiprocessing.Queue=None, port=None, proxy=None):
    asyncio.run(main.builtin(queue=queue, port=port, proxy=proxy))