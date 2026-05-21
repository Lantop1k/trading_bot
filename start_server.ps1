# Start Flask with live order execution allowed (server-side).
# Requires: MetaTrader 5 terminal installed, open, and logged in on this PC.
# Remove or set LIVE_TRADING_ENABLED=0 to block real orders.
$env:LIVE_TRADING_ENABLED = "1"
Set-Location $PSScriptRoot
python app.py
