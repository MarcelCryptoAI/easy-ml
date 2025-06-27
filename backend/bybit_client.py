from pybit.unified_trading import HTTP
from typing import List, Dict, Optional
import asyncio
import logging
from config import settings

logger = logging.getLogger(__name__)

class BybitClient:
    def __init__(self):
        self.session = HTTP(
            testnet=settings.bybit_testnet,
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
        )
    
    def get_derivatives_symbols(self) -> List[Dict]:
        try:
            response = self.session.get_instruments_info(
                category="linear"
            )
            
            if response["retCode"] == 0:
                symbols = []
                for instrument in response["result"]["list"]:
                    if instrument["status"] == "Trading":
                        symbols.append({
                            "symbol": instrument["symbol"],
                            "baseCoin": instrument["baseCoin"],
                            "quoteCoin": instrument["quoteCoin"],
                            "contractType": instrument.get("contractType", "LinearPerpetual"),
                            "leverage": instrument.get("leverageFilter", {})
                        })
                return symbols
            else:
                logger.error(f"Error fetching symbols: {response}")
                return []
        except Exception as e:
            logger.error(f"Exception in get_derivatives_symbols: {e}")
            return []
    
    def get_klines(self, symbol: str, interval: str = "1", limit: int = 200) -> List[Dict]:
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response["retCode"] == 0:
                klines = []
                for kline in response["result"]["list"]:
                    klines.append({
                        "timestamp": int(kline[0]),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5])
                    })
                return sorted(klines, key=lambda x: x["timestamp"])
            return []
        except Exception as e:
            logger.error(f"Exception in get_klines for {symbol}: {e}")
            return []
    
    def place_order(self, symbol: str, side: str, qty: float, leverage: int, 
                   take_profit: Optional[float] = None, stop_loss: Optional[float] = None,
                   order_type: str = "market", price: Optional[float] = None) -> Dict:
        try:
            # Try to set leverage but ignore if already set
            try:
                self.session.set_leverage(
                    category="linear",
                    symbol=symbol,
                    buyLeverage=str(leverage),
                    sellLeverage=str(leverage)
                )
            except Exception as leverage_error:
                # Ignore leverage errors (usually means it's already set)
                if "110043" not in str(leverage_error):
                    logger.warning(f"Leverage setting warning for {symbol}: {leverage_error}")
                pass
            
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.capitalize(),
                "orderType": "Market" if order_type.lower() == "market" else "Limit",
                "qty": str(qty),
                "timeInForce": "IOC" if order_type.lower() == "market" else "GTC"
            }
            
            # Add price for limit orders
            if order_type.lower() == "limit" and price:
                order_params["price"] = str(price)
            
            if take_profit:
                order_params["takeProfit"] = str(take_profit)
            if stop_loss:
                order_params["stopLoss"] = str(stop_loss)
            
            response = self.session.place_order(**order_params)
            
            if response["retCode"] == 0:
                return {
                    "success": True,
                    "order_id": response["result"]["orderId"],
                    "order_link_id": response["result"]["orderLinkId"]
                }
            else:
                logger.error(f"Order placement failed: {response}")
                return {"success": False, "error": response["retMsg"]}
                
        except Exception as e:
            logger.error(f"Exception in place_order: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self) -> List[Dict]:
        try:
            response = self.session.get_positions(
                category="linear",
                settleCoin="USDT"
            )
            
            if response["retCode"] == 0:
                positions = []
                for position in response["result"]["list"]:
                    if float(position["size"]) > 0:
                        positions.append({
                            "symbol": position["symbol"],
                            "side": position["side"],
                            "size": float(position["size"]),
                            "avgPrice": float(position["avgPrice"]),
                            "markPrice": float(position["markPrice"]),
                            "unrealisedPnl": float(position["unrealisedPnl"]),
                            "leverage": position["leverage"]
                        })
                return positions
            return []
        except Exception as e:
            logger.error(f"Exception in get_positions: {e}")
            return []
    
    def close_position(self, symbol: str) -> Dict:
        try:
            positions = self.get_positions()
            position = next((p for p in positions if p["symbol"] == symbol), None)
            
            if not position:
                return {"success": False, "error": "Position not found"}
            
            side = "Sell" if position["side"] == "Buy" else "Buy"
            
            response = self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(position["size"]),
                timeInForce="IOC",
                reduceOnly=True
            )
            
            if response["retCode"] == 0:
                return {
                    "success": True,
                    "order_id": response["result"]["orderId"]
                }
            else:
                return {"success": False, "error": response["retMsg"]}
                
        except Exception as e:
            logger.error(f"Exception in close_position: {e}")
            return {"success": False, "error": str(e)}