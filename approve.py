import os, time
from dotenv import load_dotenv
from web3 import Web3

load_dotenv("/root/weatherbot/.env")
PK = os.getenv("PK")
WALLET = os.getenv("WALLET")

w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
from web3.middleware import ExtraDataToPOAMiddleware
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
assert w3.is_connected(), "RPC not connected"

CHAIN_ID = 137
MAX_UINT256 = 2**256 - 1
MAX_FEE = Web3.to_wei(200, "gwei")
MAX_PRIORITY = Web3.to_wei(30, "gwei")

USDC_E = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CT = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")

SPENDERS = {
    "CTF Exchange":      Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"),
    "Neg Risk Exchange": Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a"),
    "Router":            Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"),
}

ERC20_ABI = [
    {"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],
     "name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function","stateMutability":"nonpayable"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],
     "name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function","stateMutability":"view"},
]

ERC1155_ABI = [
    {"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],
     "name":"setApprovalForAll","outputs":[],"type":"function","stateMutability":"nonpayable"},
    {"inputs":[{"name":"account","type":"address"},{"name":"operator","type":"address"}],
     "name":"isApprovedForAll","outputs":[{"name":"","type":"bool"}],"type":"function","stateMutability":"view"},
]

usdc = w3.eth.contract(address=USDC_E, abi=ERC20_ABI)
ct = w3.eth.contract(address=CT, abi=ERC1155_ABI)

tx_nonce = w3.eth.get_transaction_count(WALLET)

def send_tx(tx_data):
    global tx_nonce
    tx = {
        "from": WALLET,
        "nonce": tx_nonce,
        "chainId": CHAIN_ID,
        "maxFeePerGas": MAX_FEE,
        "maxPriorityFeePerGas": MAX_PRIORITY,
        "type": 2,
        **tx_data,
    }
    tx["gas"] = w3.eth.estimate_gas(tx)
    signed = w3.eth.account.sign_transaction(tx, PK)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  TX sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    status = "✅" if receipt["status"] == 1 else "❌ FAILED"
    print(f"  {status} | Block {receipt['blockNumber']} | Gas used {receipt['gasUsed']}")
    tx_nonce += 1
    return receipt

# --- ERC20 approvals (USDC.e) ---
print("=" * 60)
print("ERC20 APPROVE — USDC.e → max uint256")
print("=" * 60)
for name, addr in SPENDERS.items():
    print(f"\n→ Approving {name} ({addr})")
    calldata = usdc.encode_abi("approve", args=[addr, MAX_UINT256])
    send_tx({"to": USDC_E, "data": calldata})

# --- ERC1155 setApprovalForAll (Conditional Tokens) ---
print("\n" + "=" * 60)
print("ERC1155 setApprovalForAll — Conditional Tokens")
print("=" * 60)
for name, addr in SPENDERS.items():
    print(f"\n→ Approving {name} ({addr})")
    calldata = ct.encode_abi("setApprovalForAll", args=[addr, True])
    send_tx({"to": CT, "data": calldata})

# --- Verify all approvals ---
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
all_ok = True
for name, addr in SPENDERS.items():
    allowance = usdc.functions.allowance(WALLET, addr).call()
    ok = "✅" if allowance == MAX_UINT256 else "❌"
    if allowance != MAX_UINT256: all_ok = False
    print(f"USDC.e → {name}: {ok} (allowance={allowance})")

    approved = ct.functions.isApprovedForAll(WALLET, addr).call()
    ok2 = "✅" if approved else "❌"
    if not approved: all_ok = False
    print(f"CT     → {name}: {ok2} (approved={approved})")

print(f"\n{'ALL APPROVALS VERIFIED ✅' if all_ok else 'SOME APPROVALS FAILED ❌'}")
