import React, { useState } from 'react';
import {
  Container,
  Tabs,
  Tab,
  Box
} from '@mui/material';
import { Dashboard } from '../components/Dashboard';
import { CoinAnalysis } from '../components/CoinAnalysis';
import { MLProgress } from '../components/MLProgress';
import { StrategyConfigurator } from '../components/StrategyConfigurator';
import { StrategyConfig } from '../components/StrategyConfig';
import { TradingSignals } from '../components/TradingSignals';
import { TrainingStatus } from '../components/TrainingStatus';
import { StrategyOptimizer } from '../components/StrategyOptimizer';
import { ModernTradingDashboard } from '../components/ModernTradingDashboard';
import { TradingControl } from '../components/TradingControl';
import { StatusTopBar } from '../components/StatusTopBar';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

export default function Home() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <>
      {tabValue !== 0 && <StatusTopBar />}

      <Container maxWidth={false} disableGutters>
        {tabValue !== 0 && (
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="main navigation">
              <Tab label="🚀 Trading Dashboard" />
              <Tab label="🎮 Trading Control" />
              <Tab label="📊 ML Progress" />
              <Tab label="🔧 Training Status" />
              <Tab label="📡 Signals" />
              <Tab label="⚙️ Strategy Config" />
              <Tab label="🤖 AI Optimizer" />
              <Tab label="📈 Coin Analysis" />
            </Tabs>
          </Box>
        )}

        <TabPanel value={tabValue} index={0}>
          <ModernTradingDashboard onNavigate={setTabValue} />
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          <TradingControl />
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          <MLProgress />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <TrainingStatus />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <TradingSignals />
        </TabPanel>

        <TabPanel value={tabValue} index={5}>
          <StrategyConfig />
        </TabPanel>

        <TabPanel value={tabValue} index={6}>
          <StrategyOptimizer />
        </TabPanel>
        
        <TabPanel value={tabValue} index={7}>
          <CoinAnalysis />
        </TabPanel>
      </Container>
    </>
  );
}