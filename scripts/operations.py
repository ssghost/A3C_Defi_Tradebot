from brownie import accounts, config, interface, network
from web3 import Web3

def get_account():
    if network.show_active() in ["hardhat", "development", "mainnet-fork"]:
        return accounts[0]
    if network.show_active() in config["networks"]:
        account = accounts.add(config["wallets"]["from_key"])
        return account
    return None

def get_pool():
    lending_pool_addresses_provider = interface.ILendingPoolAddressesProvider(
        config["networks"][network.show_active()]["lending_pool_addresses_provider"]
    )
    lending_pool_address = lending_pool_addresses_provider.getLendingPool()
    lending_pool = interface.ILendingPool(lending_pool_address)
    return lending_pool

def approve(amount=Web3.toWei(0.1, "ether"), 
            lending_pool_address=get_pool(), 
            erc20_address=config["networks"][network.show_active()]["weth_token"], 
            account=get_account()):
    erc20 = interface.IERC20(erc20_address)
    approve_tx = erc20.approve(lending_pool_address, amount, {"from": account})
    approve_tx.wait(1)
    return True

def get_data(lending_pool=get_pool(), 
             account=get_account()):
    (
        total_collateral_eth,
        total_debt_eth,
        available_borrow_eth,
        current_liquidation_threshold,
        tlv,
        health_factor,
    ) = lending_pool.getUserAccountData(account.address)
    available_borrow_eth = Web3.fromWei(available_borrow_eth, "ether")
    total_collateral_eth = Web3.fromWei(total_collateral_eth, "ether")
    total_debt_eth = Web3.fromWei(total_debt_eth, "ether")
    return (float(total_collateral_eth), float(available_borrow_eth), float(total_debt_eth))

def get_price():
    dai_eth_price_feed = interface.AggregatorV3Interface(
        config["networks"][network.show_active()]["dai_eth_price_feed"]
    )
    latest_price = Web3.fromWei(dai_eth_price_feed.latestRoundData()[1], "ether")
    return float(latest_price)

def shift_price(amount):
    return (1 / get_price()) * (amount * 0.95)

def borrow(lending_pool=get_pool(), 
           amount=shift_price((get_data())[1]), 
           account=get_account(), 
           erc20_address=None):
    erc20_address = (
        erc20_address
        if erc20_address
        else config["networks"][network.show_active()]["aave_dai_token"]
    )
    borrow_tx = lending_pool.borrow(
        erc20_address,
        Web3.toWei(amount, "ether"),
        1,
        0,
        account.address,
        {"from": account},
    )
    borrow_tx.wait(1)
    return True

def repay(amount=shift_price((get_data())[2]), 
          lending_pool=get_pool(), 
          account=get_account()):
    approve(
        Web3.toWei(amount, "ether"),
        lending_pool,
        config["networks"][network.show_active()]["aave_dai_token"],
        account,
    )
    repay_tx = lending_pool.repay(
        config["networks"][network.show_active()]["aave_dai_token"],
        Web3.toWei(amount, "ether"),
        1,
        account.address,
        {"from": account},
    )
    repay_tx.wait(1)
    return True

def total_asset(account=get_account()):
    return float(get_data()[0])

def deposit(account=get_account(), 
            amount= Web3.toWei(0.1, "ether"),
            erc20_address = config["networks"][network.show_active()]["weth_token"],
            lending_pool=get_pool()):
    approve(amount, lending_pool.address, erc20_address, account)
    deposit_tx = lending_pool.deposit(erc20_address, amount, account.address, 0, {"from": account})
    deposit_tx.wait(1)
    return True



