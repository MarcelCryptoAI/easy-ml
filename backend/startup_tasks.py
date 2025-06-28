import asyncio
import logging
from sqlalchemy.orm import Session
from database import SessionLocal, create_tables, Coin, TradingStrategy
from bybit_client import BybitClient
from config import settings
from allowed_coins import ALLOWED_COINS

logger = logging.getLogger(__name__)

async def initialize_database():
    """Initialize database tables and default data"""
    try:
        create_tables()
        logger.info("Database tables created successfully")
        
        # Initialize default strategies for coins
        db = SessionLocal()
        
        # Check if we have any coins, if not sync them
        coin_count = db.query(Coin).count()
        if coin_count == 0:
            await sync_initial_coins(db)
        else:
            # Clean up coins not in allowed list
            await cleanup_disallowed_coins(db)
        
        # Ensure all coins have default strategies
        coins = db.query(Coin).filter(Coin.is_active == True).all()
        for coin in coins:
            existing_strategy = db.query(TradingStrategy).filter(
                TradingStrategy.coin_symbol == coin.symbol,
                TradingStrategy.is_active == True
            ).first()
            
            if not existing_strategy:
                default_strategy = TradingStrategy(
                    coin_symbol=coin.symbol,
                    take_profit_percentage=settings.take_profit_percentage,
                    stop_loss_percentage=settings.stop_loss_percentage,
                    leverage=10,
                    position_size_percentage=5.0,
                    confidence_threshold=settings.confidence_threshold
                )
                db.add(default_strategy)
        
        db.commit()
        db.close()
        logger.info(f"Initialized strategies for {len(coins)} coins")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

async def sync_initial_coins(db: Session):
    """Sync initial coins from Bybit"""
    bybit_client = BybitClient()
    symbols = bybit_client.get_derivatives_symbols()
    
    if not symbols:
        raise Exception("Failed to fetch coins from Bybit API - check API credentials")
    
    for symbol_data in symbols:
        existing_coin = db.query(Coin).filter(Coin.symbol == symbol_data["symbol"]).first()
        if not existing_coin:
            new_coin = Coin(
                symbol=symbol_data["symbol"],
                name=f"{symbol_data['baseCoin']}/{symbol_data['quoteCoin']}"
            )
            db.add(new_coin)
    
    db.commit()
    logger.info(f"Synced {len(symbols)} coins from Bybit")

async def cleanup_disallowed_coins(db: Session):
    """Remove coins that are not in the allowed list"""
    allowed_set = set(ALLOWED_COINS)
    
    # Get all coins not in allowed list
    disallowed_coins = db.query(Coin).filter(~Coin.symbol.in_(ALLOWED_COINS)).all()
    
    if disallowed_coins:
        for coin in disallowed_coins:
            # Delete associated strategies first
            db.query(TradingStrategy).filter(TradingStrategy.coin_symbol == coin.symbol).delete()
            # Delete the coin
            db.delete(coin)
        
        db.commit()
        logger.info(f"Removed {len(disallowed_coins)} coins not in allowed list")

def validate_configuration():
    """Validate that all required configuration is present"""
    import os
    missing = []
    
    # Check all required environment variables
    required_vars = {
        "BYBIT_API_KEY": settings.bybit_api_key or os.getenv("BYBIT_API_KEY"),
        "BYBIT_API_SECRET": settings.bybit_api_secret or os.getenv("BYBIT_API_SECRET"),
        "OPENAI_API_KEY": settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        "DATABASE_URL": settings.database_url or os.getenv("DATABASE_URL")
    }
    
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing.append(var_name)
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Please set these variables in Railway dashboard under 'Variables' tab")
        
        # Show what Railway should have automatically
        if "DATABASE_URL" in missing:
            logger.error("DATABASE_URL should be automatically provided by Railway PostgreSQL")
            logger.error("Make sure PostgreSQL plugin is added to your Railway project")
        
        raise Exception(f"Missing required environment variables: {', '.join(missing)}")
    
    logger.info("Configuration validation passed - all required keys present")
    logger.info(f"Using database: {required_vars['DATABASE_URL'][:20]}...")  # Show partial URL