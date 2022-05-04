from brownie import accounts, config, interface, network
from web3 import Web3

def get_account():
    if network.show_active() in ["hardhat", "development", "mainnet-fork"]:
        return accounts[0]
    if network.show_active() in config["networks"]:
        account = accounts.add(config["wallets"]["from_key"])
        return account
    return None

def get_lending_pool():
    lending_pool_addresses_provider = interface.ILendingPoolAddressesProvider(
        config["networks"][network.show_active()]["lending_pool_addresses_provider"]
    )
    lending_pool_address = lending_pool_addresses_provider.getLendingPool()
    lending_pool = interface.ILendingPool(lending_pool_address)
    return lending_pool

def approve_erc20(amount=Web3.toWei(config["fee"], "ether"), 
                  lending_pool_address=get_lending_pool(), 
                  erc20_address=config["networks"][network.show_active()]["weth_token"], 
                  account=get_account()):
    print("Approving ERC20...")
    erc20 = interface.IERC20(erc20_address)
    tx_hash = erc20.approve(lending_pool_address, amount, {"from": account})
    tx_hash.wait(1)
    print("Approved!")
    return True




