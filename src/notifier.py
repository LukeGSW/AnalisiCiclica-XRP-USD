"""
Notification module for Kriterion Quant Trading System
Handles Telegram notifications for trading signals
"""

import requests
import json
from typing import Dict, Optional
from datetime import datetime

from config import Config

class TelegramNotifier:
    """Class to handle Telegram notifications"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize the Telegram notifier
        
        Parameters
        ----------
        bot_token : str, optional
            Telegram bot token. Defaults to Config.TELEGRAM_BOT_TOKEN
        chat_id : str, optional
            Telegram chat ID. Defaults to Config.TELEGRAM_CHAT_ID
        """
        self.bot_token = bot_token or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        
        if not self.bot_token or not self.chat_id:
            print("⚠️ Telegram credentials not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a text message via Telegram
        
        Parameters
        ----------
        message : str
            Message to send
        parse_mode : str, optional
            Parse mode for formatting. Default is "Markdown"
        
        Returns
        -------
        bool
            True if message sent successfully
        """
        if not self.enabled:
            print("❌ Telegram notifications are disabled")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("✅ Telegram notification sent successfully")
                return True
            else:
                print(f"❌ Failed to send Telegram notification: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending Telegram notification: {e}")
            return False
    
    def send_signal_alert(self, signal_info: Dict) -> bool:
        """
        Send a formatted signal alert
        
        Parameters
        ----------
        signal_info : Dict
            Signal information dictionary
        
        Returns
        -------
        bool
            True if alert sent successfully
        """
        if not self.enabled:
            return False
        
        # Format the message
        emoji_map = {
            'BUY': '🟢',
            'SELL': '🔴', 
            'HOLD': '⏸️'
        }
        
        confidence_emoji = {
            'HIGH': '⭐⭐⭐',
            'MEDIUM': '⭐⭐',
            'LOW': '⭐'
        }
        
        message = f"""
{emoji_map.get(signal_info['signal'], '❓')} *SIGNAL ALERT - {Config.TICKER}*
{'='*30}

📅 *Date:* {signal_info['date']}
📊 *Signal:* `{signal_info['signal']}`
💰 *Price:* ${signal_info['price']:.2f}
📈 *Position:* {signal_info['position']}

*Cycle Analysis:*
🔄 Phase: {signal_info['phase_quadrant']}
📐 Phase Value: {signal_info['phase_value']:.2f} rad
📊 Oscillator: {signal_info['oscillator_value']:.4f}

*Signal Quality:*
💪 Strength: {signal_info['signal_strength']:.1f}/100
{confidence_emoji.get(signal_info['confidence'], '')} Confidence: {signal_info['confidence']}

⏰ Generated: {datetime.now().strftime('%H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def send_backtest_summary(self, metrics: Dict) -> bool:
        """
        Send a backtest performance summary
        
        Parameters
        ----------
        metrics : Dict
            Backtest metrics dictionary
        
        Returns
        -------
        bool
            True if summary sent successfully
        """
        if not self.enabled:
            return False
        
        message = f"""
📊 *BACKTEST SUMMARY - {Config.TICKER}*
{'='*30}

*Performance Metrics:*
📈 Total Return: {metrics.get('total_return_%', 0):.2f}%
📉 Max Drawdown: {metrics.get('max_drawdown_%', 0):.2f}%
📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
🎯 Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
💹 Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}

*Trading Statistics:*
🔄 Total Trades: {metrics.get('total_trades', 0)}
✅ Win Rate: {metrics.get('win_rate_%', 0):.1f}%
💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}

⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def send_error_alert(self, error_message: str) -> bool:
        """
        Send an error alert
        
        Parameters
        ----------
        error_message : str
            Error message to send
        
        Returns
        -------
        bool
            True if alert sent successfully
        """
        if not self.enabled:
            return False
        
        message = f"""
🚨 *ERROR ALERT - {Config.TICKER}*
{'='*30}

❌ An error occurred in the trading system:

`{error_message}`

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the system logs for more details.
"""
        
        return self.send_message(message)
    
    def send_daily_summary(self, signal_info: Dict, metrics: Dict) -> bool:
        """
        Send a comprehensive daily summary
        
        Parameters
        ----------
        signal_info : Dict
            Latest signal information
        metrics : Dict
            Latest backtest metrics
        
        Returns
        -------
        bool
            True if summary sent successfully
        """
        if not self.enabled:
            return False
        
        # Determine action emoji
        action_emoji = "🟢 LONG" if signal_info['position'] == 'LONG' else "⏸️ FLAT"
        
        message = f"""
📰 *DAILY SUMMARY - {Config.TICKER}*
{'='*30}

*Current Status:*
{action_emoji} Position: {signal_info['position']}
💰 Price: ${signal_info['price']:.2f}
📊 Last Signal: {signal_info['signal']}
📅 Date: {signal_info['date']}

*Cycle Status:*
🔄 Phase: {signal_info['phase_quadrant']}
📐 Phase: {signal_info['phase_value']:.2f} rad
💪 Signal Strength: {signal_info['signal_strength']:.1f}/100

*Recent Performance:*
📈 Strategy Return: {metrics.get('total_return_%', 0):.2f}%
📉 Max Drawdown: {metrics.get('max_drawdown_%', 0):.2f}%
📊 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
✅ Win Rate: {metrics.get('win_rate_%', 0):.1f}%

*Next Steps:*
{"🎯 Monitor for exit signal" if signal_info['position'] == 'LONG' else "👀 Waiting for entry signal"}

⏰ Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message)
    
    def test_connection(self) -> bool:
        """
        Test the Telegram connection
        
        Returns
        -------
        bool
            True if connection successful
        """
        if not self.enabled:
            print("❌ Telegram notifications are disabled")
            return False
        
        test_message = f"""
✅ *Connection Test Successful*

Bot is connected and ready to send notifications for {Config.TICKER}.

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(test_message)
