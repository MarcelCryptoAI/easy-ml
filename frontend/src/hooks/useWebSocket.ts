import { useEffect, useState, useRef } from 'react';
import { io, Socket } from 'socket.io-client';
import { WebSocketMessage } from '../types';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

export const useWebSocket = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const messageHandlers = useRef<{ [key: string]: (data: any) => void }>({});

  useEffect(() => {
    const socketConnection = io(WS_URL, {
      transports: ['websocket']
    });

    socketConnection.on('connect', () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    });

    socketConnection.on('disconnect', () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    });

    socketConnection.on('message', (data: string) => {
      try {
        const message: WebSocketMessage = JSON.parse(data);
        setMessages(prev => [...prev.slice(-99), message]); // Keep last 100 messages
        
        const handler = messageHandlers.current[message.type];
        if (handler) {
          handler(message.data);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    setSocket(socketConnection);

    return () => {
      socketConnection.disconnect();
    };
  }, []);

  const subscribe = (messageType: string, handler: (data: any) => void) => {
    messageHandlers.current[messageType] = handler;
  };

  const unsubscribe = (messageType: string) => {
    delete messageHandlers.current[messageType];
  };

  const sendMessage = (message: any) => {
    if (socket && isConnected) {
      socket.emit('message', JSON.stringify(message));
    }
  };

  return {
    socket,
    isConnected,
    messages,
    subscribe,
    unsubscribe,
    sendMessage
  };
};