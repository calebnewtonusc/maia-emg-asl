import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, TextInput, Switch, TouchableOpacity,
  ScrollView, StyleSheet, Alert, ActivityIndicator,
} from 'react-native';
import { getServerConfig, updateServerConfig } from '../config/serverConfig';

interface ModelInfo {
  n_classes: number;
  classes: string[];
  sample_rate: number;
  confidence_threshold: number;
}

export default function SettingsScreen() {
  const cfg = getServerConfig();
  const [serverUrl, setServerUrl] = useState(cfg.serverUrl);
  const [apiKey, setApiKey] = useState(cfg.apiKey);
  const [onDeviceOnly, setOnDeviceOnly] = useState(cfg.onDeviceOnly);
  const [fallbackToServer, setFallbackToServer] = useState(cfg.fallbackToServer);
  const [testing, setTesting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'ok' | 'error'>('idle');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  const handleSave = useCallback(() => {
    updateServerConfig({ serverUrl, apiKey, onDeviceOnly, fallbackToServer });
    Alert.alert('Saved', 'Settings updated successfully');
  }, [serverUrl, apiKey, onDeviceOnly, fallbackToServer]);

  const handleTestConnection = useCallback(async () => {
    setTesting(true);
    setConnectionStatus('idle');
    setModelInfo(null);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      const resp = await fetch(`${serverUrl}/health`, {
        headers: { 'X-API-Key': apiKey },
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (resp.ok) {
        setConnectionStatus('ok');
        // Fetch model info
        try {
          const infoResp = await fetch(`${serverUrl}/info`, {
            headers: { 'X-API-Key': apiKey },
          });
          if (infoResp.ok) {
            setModelInfo(await infoResp.json());
          }
        } catch { /* ignore */ }
      } else {
        setConnectionStatus('error');
      }
    } catch {
      setConnectionStatus('error');
    } finally {
      setTesting(false);
    }
  }, [serverUrl, apiKey]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Server Config */}
      <Text style={styles.sectionHeader}>Server Configuration</Text>
      <View style={styles.card}>
        <Text style={styles.label}>Server URL</Text>
        <TextInput
          style={styles.input}
          value={serverUrl}
          onChangeText={setServerUrl}
          placeholder="https://maia.railway.app"
          placeholderTextColor="#666"
          autoCapitalize="none"
          keyboardType="url"
        />
        <Text style={styles.label}>API Key</Text>
        <TextInput
          style={styles.input}
          value={apiKey}
          onChangeText={setApiKey}
          placeholder="your-api-key"
          placeholderTextColor="#666"
          secureTextEntry
          autoCapitalize="none"
        />
        <TouchableOpacity
          style={[styles.button, testing && styles.buttonDisabled]}
          onPress={handleTestConnection}
          disabled={testing}
        >
          {testing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>Test Connection</Text>
          )}
        </TouchableOpacity>

        {connectionStatus === 'ok' && (
          <Text style={[styles.statusText, styles.statusOk]}>Connected</Text>
        )}
        {connectionStatus === 'error' && (
          <Text style={[styles.statusText, styles.statusError]}>Connection failed</Text>
        )}

        {modelInfo && (
          <View style={styles.infoBox}>
            <Text style={styles.infoText}>Classes: {modelInfo.n_classes}</Text>
            <Text style={styles.infoText}>Sample rate: {modelInfo.sample_rate} Hz</Text>
            <Text style={styles.infoText}>Confidence threshold: {modelInfo.confidence_threshold}</Text>
          </View>
        )}
      </View>

      {/* Inference Mode */}
      <Text style={styles.sectionHeader}>Inference Mode</Text>
      <View style={styles.card}>
        <View style={styles.row}>
          <View style={styles.rowText}>
            <Text style={styles.label}>On-Device Only</Text>
            <Text style={styles.sublabel}>Use ONNX model on device, no network required</Text>
          </View>
          <Switch
            value={onDeviceOnly}
            onValueChange={setOnDeviceOnly}
            trackColor={{ false: '#3d3d3d', true: '#4CAF50' }}
            thumbColor="#fff"
          />
        </View>
        <View style={styles.divider} />
        <View style={styles.row}>
          <View style={styles.rowText}>
            <Text style={styles.label}>Fallback to Server</Text>
            <Text style={styles.sublabel}>Use Railway server when on-device confidence is low</Text>
          </View>
          <Switch
            value={fallbackToServer}
            onValueChange={setFallbackToServer}
            disabled={onDeviceOnly}
            trackColor={{ false: '#3d3d3d', true: '#4CAF50' }}
            thumbColor="#fff"
          />
        </View>
      </View>

      {/* Save */}
      <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
        <Text style={styles.saveButtonText}>Save Settings</Text>
      </TouchableOpacity>

      {/* About */}
      <Text style={styles.sectionHeader}>About</Text>
      <View style={styles.card}>
        <Text style={styles.infoText}>MAIA EMG-ASL v0.1.0</Text>
        <Text style={styles.infoText}>MAIA Biotech — Real-time ASL from sEMG</Text>
        <Text style={styles.sublabel}>8-channel wrist neural band, 200 Hz, 26 ASL classes</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#121212' },
  content: { padding: 16, paddingBottom: 48 },
  sectionHeader: {
    color: '#aaa', fontSize: 12, fontWeight: '600',
    textTransform: 'uppercase', letterSpacing: 1,
    marginTop: 24, marginBottom: 8, marginLeft: 4,
  },
  card: {
    backgroundColor: '#1e1e1e', borderRadius: 12,
    padding: 16, marginBottom: 8,
  },
  label: { color: '#fff', fontSize: 14, fontWeight: '500', marginBottom: 6 },
  sublabel: { color: '#888', fontSize: 12, marginTop: 2 },
  input: {
    backgroundColor: '#2a2a2a', color: '#fff',
    borderRadius: 8, padding: 12, marginBottom: 12,
    fontSize: 14, borderWidth: 1, borderColor: '#333',
  },
  button: {
    backgroundColor: '#4CAF50', borderRadius: 8,
    padding: 12, alignItems: 'center', marginTop: 4,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 14 },
  statusText: { textAlign: 'center', marginTop: 8, fontWeight: '500' },
  statusOk: { color: '#4CAF50' },
  statusError: { color: '#f44336' },
  infoBox: {
    backgroundColor: '#2a2a2a', borderRadius: 8,
    padding: 12, marginTop: 12,
  },
  infoText: { color: '#ccc', fontSize: 13, marginBottom: 4 },
  row: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between', paddingVertical: 4,
  },
  rowText: { flex: 1, paddingRight: 16 },
  divider: { height: 1, backgroundColor: '#333', marginVertical: 12 },
  saveButton: {
    backgroundColor: '#2196F3', borderRadius: 12,
    padding: 16, alignItems: 'center', marginTop: 16, marginBottom: 8,
  },
  saveButtonText: { color: '#fff', fontWeight: '700', fontSize: 16 },
});
