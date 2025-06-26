import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  IconButton,
  Button,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import { Refresh, ExpandMore, PlayArrow, Pause, Settings } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { tradingApi } from '../utils/api';
import toast from 'react-hot-toast';

interface TrainingQueueItem {
  coin_symbol: string;
  model_type: string;
  status: 'pending' | 'training' | 'completed' | 'error';
  progress: number;
  estimated_time_remaining: number;
  started_at?: string;
  completed_at?: string;
  queue_position: number;
}

interface TrainingSession {
  current_coin: string;
  current_model: string;
  progress: number;
  eta_seconds: number;
  total_queue_items: number;
  completed_items: number;
  session_start_time: string;
  estimated_completion_time: string;
}

export const TrainingStatus: React.FC = () => {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const queryClient = useQueryClient();

  const { data: trainingSession, isLoading: sessionLoading, refetch: refetchSession } = useQuery({
    queryKey: ['training-session'],
    queryFn: async () => {
      // Mock training session data - replace with actual API call
      const mockSession: TrainingSession = {
        current_coin: 'BTCUSDT',
        current_model: 'LSTM',
        progress: 67,
        eta_seconds: 450,
        total_queue_items: 2000, // 500 coins × 4 models
        completed_items: 1340,
        session_start_time: new Date(Date.now() - 3600000).toISOString(),
        estimated_completion_time: new Date(Date.now() + 86400000).toISOString()
      };
      return mockSession;
    },
    refetchInterval: autoRefresh ? 5000 : false
  });

  const { data: trainingQueue = [], isLoading: queueLoading, refetch: refetchQueue } = useQuery({
    queryKey: ['training-queue'],
    queryFn: async () => {
      // Mock queue data - replace with actual API call
      const mockQueue: TrainingQueueItem[] = [
        {
          coin_symbol: 'BTCUSDT',
          model_type: 'LSTM',
          status: 'training',
          progress: 67,
          estimated_time_remaining: 450,
          started_at: new Date(Date.now() - 300000).toISOString(),
          queue_position: 1
        },
        {
          coin_symbol: 'BTCUSDT',
          model_type: 'Random Forest',
          status: 'pending',
          progress: 0,
          estimated_time_remaining: 180,
          queue_position: 2
        },
        {
          coin_symbol: 'BTCUSDT',
          model_type: 'SVM',
          status: 'pending',
          progress: 0,
          estimated_time_remaining: 120,
          queue_position: 3
        },
        {
          coin_symbol: 'BTCUSDT',
          model_type: 'Neural Network',
          status: 'pending',
          progress: 0,
          estimated_time_remaining: 300,
          queue_position: 4
        },
        {
          coin_symbol: 'ETHUSDT',
          model_type: 'LSTM',
          status: 'pending',
          progress: 0,
          estimated_time_remaining: 420,
          queue_position: 5
        }
      ];
      return mockQueue;
    },
    refetchInterval: autoRefresh ? 3000 : false
  });

  const pauseTrainingMutation = useMutation({
    mutationFn: async () => {
      // API call to pause training
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/pause`, {
        method: 'POST'
      });
      return response.json();
    },
    onSuccess: () => {
      toast.success('Training paused');
      refetchSession();
    }
  });

  const resumeTrainingMutation = useMutation({
    mutationFn: async () => {
      // API call to resume training
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/resume`, {
        method: 'POST'
      });
      return response.json();
    },
    onSuccess: () => {
      toast.success('Training resumed');
      refetchSession();
    }
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'training': return 'warning';
      case 'completed': return 'success';
      case 'pending': return 'default';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const formatEstimatedCompletion = (isoString: string) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) {
      return `${diffDays} days, ${diffHours % 24} hours`;
    } else {
      return `${diffHours} hours`;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">🤖 ML Training Status</Typography>
        <Box display="flex" gap={1}>
          <Button
            variant="outlined"
            color={autoRefresh ? 'success' : 'default'}
            onClick={() => setAutoRefresh(!autoRefresh)}
            startIcon={autoRefresh ? <Pause /> : <PlayArrow />}
          >
            {autoRefresh ? 'Auto' : 'Manual'}
          </Button>
          <IconButton onClick={() => { refetchSession(); refetchQueue(); }}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Current Training Session Overview */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Training Session
              </Typography>
              {trainingSession && (
                <>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">
                      Training: {trainingSession.current_coin} - {trainingSession.current_model}
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={trainingSession.progress} 
                      sx={{ mt: 1, height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="caption" color="textSecondary">
                      {trainingSession.progress}% - ETA: {formatTime(trainingSession.eta_seconds)}
                    </Typography>
                  </Box>
                  
                  <Typography variant="body2" color="textSecondary">
                    Queue Progress: {trainingSession.completed_items}/{trainingSession.total_queue_items}
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={(trainingSession.completed_items / trainingSession.total_queue_items) * 100} 
                    sx={{ mt: 1, mb: 2 }}
                  />
                  
                  <Typography variant="caption" display="block" color="textSecondary">
                    Session started: {new Date(trainingSession.session_start_time).toLocaleString()}
                  </Typography>
                  <Typography variant="caption" display="block" color="textSecondary">
                    Estimated completion: {formatEstimatedCompletion(trainingSession.estimated_completion_time)}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Controls
              </Typography>
              <Box display="flex" gap={1} flexWrap="wrap">
                <Button
                  variant="contained"
                  color="warning"
                  startIcon={<Pause />}
                  onClick={() => pauseTrainingMutation.mutate()}
                  disabled={pauseTrainingMutation.isPending}
                >
                  Pause Training
                </Button>
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<PlayArrow />}
                  onClick={() => resumeTrainingMutation.mutate()}
                  disabled={resumeTrainingMutation.isPending}
                >
                  Resume Training
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Settings />}
                  onClick={() => toast.info('Training settings coming soon')}
                >
                  Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Training Queue */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6">
            Training Queue ({trainingQueue.length} items)
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          {queueLoading ? (
            <Alert severity="info">Loading training queue...</Alert>
          ) : trainingQueue.length > 0 ? (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Position</TableCell>
                    <TableCell>Coin</TableCell>
                    <TableCell>Model Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>ETA</TableCell>
                    <TableCell>Started</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trainingQueue.slice(0, 20).map((item, index) => (
                    <TableRow key={`${item.coin_symbol}-${item.model_type}`}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          #{item.queue_position}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="subtitle2" fontWeight="bold">
                          {item.coin_symbol}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={item.model_type}
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={item.status}
                          color={getStatusColor(item.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box width={100}>
                          <LinearProgress 
                            variant="determinate" 
                            value={item.progress}
                            sx={{ mb: 0.5 }}
                          />
                          <Typography variant="caption">
                            {item.progress}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {item.estimated_time_remaining > 0 ? formatTime(item.estimated_time_remaining) : '-'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption">
                          {item.started_at ? new Date(item.started_at).toLocaleTimeString() : '-'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Alert severity="info">No training items in queue</Alert>
          )}
          
          {trainingQueue.length > 20 && (
            <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
              Showing first 20 items. {trainingQueue.length - 20} more items in queue.
            </Typography>
          )}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};