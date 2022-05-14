from models.A3C_model import *

async def main():
    A3c = A3CAgent()
    await A3c.train_with_threads()
    await A3c.test()
    await A3c.save()
