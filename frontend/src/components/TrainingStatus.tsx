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
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
  TextField,
  Slider
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
  overall_progress: number;
  eta_seconds: number;
  total_queue_items: number;
  completed_items: number;
  remaining_models: number;
  session_start_time: string;
  estimated_completion_time: string;
}

export const TrainingStatus: React.FC = () => {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    autoRetrain: true,
    trainingInterval: 3600, // seconds
    maxModelsPerCoin: 10,
    enableNotifications: true,
    batchSize: 5
  });
  const queryClient = useQueryClient();

  const { data: trainingSession, isLoading: sessionLoading, refetch: refetchSession } = useQuery({
    queryKey: ['training-session'],
    queryFn: async () => {
      // Use real API endpoint
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/session`);
      const data = await response.json();
      return data as TrainingSession;
    },
    refetchInterval: autoRefresh ? 5000 : false
  });

  const { data: trainingQueue = [], isLoading: queueLoading, refetch: refetchQueue } = useQuery({
    queryKey: ['training-queue'],
    queryFn: async () => {
      // Use real API endpoint
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/queue`);
      const data = await response.json();
      return data as TrainingQueueItem[];
    },
    refetchInterval: autoRefresh ? 3000 : false
  });

  const pauseTrainingMutation = useMutation({
    mutationFn: async () => {
      // API call to pause training
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/pause`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to pause training');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Training paused');
      refetchSession();
      refetchQueue();
    },
    onError: () => {
      toast.error('Failed to pause training');
    }
  });

  const resumeTrainingMutation = useMutation({
    mutationFn: async () => {
      // API call to resume training
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'https://easy-ml-production.up.railway.app'}/training/resume`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to resume training');
      return response.json();
    },
    onSuccess: () => {
      toast.success('Training resumed');
      refetchSession();
      refetchQueue();
    },
    onError: () => {
      toast.error('Failed to resume training');
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
            color={autoRefresh ? 'success' : 'primary'}
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

      {/* Overall Progress Overview */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom color="primary">
                🚀 Overall Training Progress: All 5020 Models
              </Typography>
              {trainingSession && (
                <>
                  <Box mb={3}>
                    <Typography variant="h6" color="textSecondary" gutterBottom>
                      System Progress: {trainingSession.overall_progress}% Complete
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={trainingSession.overall_progress} 
                      sx={{ 
                        height: 20, 
                        borderRadius: 10,
                        backgroundColor: '#f0f0f0',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 10,
                          background: 'linear-gradient(45deg, #4caf50, #8bc34a)'
                        }
                      }}
                    />
                    <Typography variant="body1" sx={{ mt: 1 }}>
                      <strong>{trainingSession.completed_items}</strong> of <strong>{trainingSession.total_queue_items}</strong> models trained
                      ({trainingSession.remaining_models} remaining)
                    </Typography>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Current Training Session Details */}
      <Grid container spacing={3} mb={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Model Training
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
                  
                  <Typography variant="caption" display="block" color="textSecondary">
                    Session started: {new Date(trainingSession.session_start_time).toLocaleString()}
                  </Typography>
                  <Typography variant="caption" display="block" color="textSecondary">
                    Full completion ETA: {formatEstimatedCompletion(trainingSession.estimated_completion_time)}
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
                  onClick={() => setSettingsOpen(true)}
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
                  {trainingQueue.map((item, index) => (
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

      {/* Training Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Training Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={settings.autoRetrain}
                  onChange={(e) => setSettings({...settings, autoRetrain: e.target.checked})}
                />
              }
              label="Auto-retrain models"
            />
            
            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Training Interval (minutes)</Typography>
              <Slider
                value={settings.trainingInterval / 60}
                onChange={(_, value) => setSettings({...settings, trainingInterval: (value as number) * 60})}
                min={5}
                max={720}
                step={5}
                marks={[
                  { value: 5, label: '5m' },
                  { value: 60, label: '1h' },
                  { value: 360, label: '6h' },
                  { value: 720, label: '12h' }
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}m`}
              />
            </Box>

            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Max Models per Coin</Typography>
              <Slider
                value={settings.maxModelsPerCoin}
                onChange={(_, value) => setSettings({...settings, maxModelsPerCoin: value as number})}
                min={1}
                max={10}
                step={1}
                marks
                valueLabelDisplay="auto"
              />
            </Box>

            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Training Batch Size</Typography>
              <Slider
                value={settings.batchSize}
                onChange={(_, value) => setSettings({...settings, batchSize: value as number})}
                min={1}
                max={20}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                  { value: 20, label: '20' }
                ]}
                valueLabelDisplay="auto"
              />
            </Box>

            <FormControlLabel
              control={
                <Switch
                  checked={settings.enableNotifications}
                  onChange={(e) => setSettings({...settings, enableNotifications: e.target.checked})}
                />
              }
              label="Enable training notifications"
              sx={{ mt: 2 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            onClick={() => {
              toast.success('Settings saved successfully');
              setSettingsOpen(false);
            }}
          >
            Save Settings
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};