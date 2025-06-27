#!/usr/bin/env python3

import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
import pandas as pd
from database import HistoricalData, Coin, get_db
from bybit_client import BybitClient

logger = logging.getLogger(__name__)

class HistoricalDataService:
    """Service for fetching and storing 6 months of historical data"""
    
    def __init__(self):
        self.bybit_client = BybitClient()
        
    async def fetch_and_store_historical_data(self, coin_symbol: str, timeframe: str = "1h") -> bool:
        """
        Fetch 6 months of historical data for a coin and store in database
        
        Args:
            coin_symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for data (1h, 4h, 1d)
            
        Returns:
            bool: Success status
        """
        try:
            db = next(get_db())
            
            # Check if we already have recent data
            existing_data = db.query(HistoricalData).filter(
                HistoricalData.coin_symbol == coin_symbol,
                HistoricalData.timeframe == timeframe
            ).order_by(HistoricalData.timestamp.desc()).first()
            
            # Calculate date range - 6 months back
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=180)  # 6 months
            
            if existing_data and existing_data.timestamp > (datetime.utcnow() - timedelta(days=1)):
                logger.info(f"Recent data already exists for {coin_symbol}, skipping...")
                return True
            
            logger.info(f"🔄 Fetching 6 months historical data for {coin_symbol} ({timeframe})")
            
            # Fetch data from Bybit in chunks (Bybit has limits)
            all_klines = []
            current_end = end_time
            
            while current_end > start_time:
                # Calculate chunk start (max 200 candles per request)
                if timeframe == "1h":
                    chunk_start = current_end - timedelta(hours=200)
                elif timeframe == "4h":
                    chunk_start = current_end - timedelta(hours=800)  # 200 * 4
                elif timeframe == "1d":
                    chunk_start = current_end - timedelta(days=200)
                else:
                    chunk_start = current_end - timedelta(hours=200)
                
                if chunk_start < start_time:
                    chunk_start = start_time
                
                try:
                    # Get kline data from Bybit
                    klines = self.bybit_client.session.get_kline(
                        category="linear",
                        symbol=coin_symbol,
                        interval=timeframe,
                        start=int(chunk_start.timestamp() * 1000),
                        end=int(current_end.timestamp() * 1000),
                        limit=200
                    )
                    
                    if klines and klines.get("result") and klines["result"].get("list"):
                        all_klines.extend(klines["result"]["list"])
                        logger.info(f"📊 Fetched {len(klines['result']['list'])} candles for {coin_symbol}")
                    
                    current_end = chunk_start
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)  # 2 requests per second
                    
                except Exception as e:
                    logger.error(f"Error fetching chunk for {coin_symbol}: {e}")
                    current_end = chunk_start
                    await asyncio.sleep(2)  # Longer wait on error
            
            if not all_klines:
                logger.warning(f"No historical data retrieved for {coin_symbol}")
                return False
            
            # Process and store data
            stored_count = 0
            for kline in all_klines:
                try:
                    # Bybit kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
                    timestamp = datetime.fromtimestamp(int(kline[0]) / 1000)
                    
                    # Check if this data point already exists
                    existing = db.query(HistoricalData).filter(
                        HistoricalData.coin_symbol == coin_symbol,
                        HistoricalData.timestamp == timestamp,
                        HistoricalData.timeframe == timeframe
                    ).first()
                    
                    if existing:
                        continue
                    
                    historical_record = HistoricalData(
                        coin_symbol=coin_symbol,
                        timestamp=timestamp,
                        open_price=float(kline[1]),
                        high_price=float(kline[2]),
                        low_price=float(kline[3]),
                        close_price=float(kline[4]),
                        volume=float(kline[5]),
                        timeframe=timeframe
                    )
                    
                    db.add(historical_record)
                    stored_count += 1
                    
                    # Commit in batches
                    if stored_count % 100 == 0:
                        db.commit()
                        
                except Exception as e:
                    logger.error(f"Error processing kline for {coin_symbol}: {e}")
                    continue
            
            # Final commit
            db.commit()
            logger.info(f"✅ Stored {stored_count} historical records for {coin_symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in fetch_and_store_historical_data for {coin_symbol}: {e}")
            return False
        finally:
            db.close()
    
    async def fetch_all_coins_historical_data(self):
        """Fetch 6 months historical data for all active coins"""
        try:
            db = next(get_db())
            active_coins = db.query(Coin).filter(Coin.is_active == True).all()
            
            logger.info(f"🚀 Starting historical data fetch for {len(active_coins)} coins")
            
            success_count = 0
            for i, coin in enumerate(active_coins):
                try:
                    logger.info(f"📈 Processing {coin.symbol} ({i+1}/{len(active_coins)})")
                    
                    # Fetch 1h timeframe data
                    success = await self.fetch_and_store_historical_data(coin.symbol, "1h")
                    if success:
                        success_count += 1
                    
                    # Rate limiting between coins
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing historical data for {coin.symbol}: {e}")
                    continue
            
            logger.info(f"✅ Historical data fetch completed: {success_count}/{len(active_coins)} coins successful")
            
        except Exception as e:
            logger.error(f"Error in fetch_all_coins_historical_data: {e}")
        finally:
            db.close()
    
    def get_historical_data(self, db: Session, coin_symbol: str, timeframe: str = "1h", days: int = 30) -> List[Dict]:
        """
        Get historical data for a coin from database
        
        Args:
            db: Database session
            coin_symbol: Trading pair symbol
            timeframe: Timeframe for data
            days: Number of days to retrieve
            
        Returns:
            List of historical data records
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            records = db.query(HistoricalData).filter(
                HistoricalData.coin_symbol == coin_symbol,
                HistoricalData.timeframe == timeframe,
                HistoricalData.timestamp >= cutoff_date
            ).order_by(HistoricalData.timestamp.asc()).all()
            
            return [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "open": record.open_price,
                    "high": record.high_price,
                    "low": record.low_price,
                    "close": record.close_price,
                    "volume": record.volume
                }
                for record in records
            ]
            
        except Exception as e:
            logger.error(f"Error getting historical data for {coin_symbol}: {e}")
            return []

# Global instance
historical_service = HistoricalDataService()