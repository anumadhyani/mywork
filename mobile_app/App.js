import { StatusBar } from 'expo-status-bar';
import Constants from 'expo-constants';
import * as ImagePicker from 'expo-image-picker';
import { useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

export default function App() {
  const apiBaseUrl = useMemo(() => {
    const extra = Constants?.expoConfig?.extra;
    const url = (extra?.apiBaseUrl || extra?.API_BASE_URL || '').toString().trim();
    return url.replace(/\/$/, '');
  }, []);

  const [apiKey, setApiKey] = useState('');
  const [imageAsset, setImageAsset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const pickImage = async () => {
    setError('');
    setResult(null);

    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setError('Gallery permission is required.');
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (res.canceled) return;
    const asset = res.assets?.[0] || null;
    setImageAsset(asset);
  };

  const analyze = async () => {
    setError('');
    setResult(null);

    if (!apiBaseUrl) {
      setError('Missing apiBaseUrl. Set it in mobile_app/app.json (expo.extra.apiBaseUrl).');
      return;
    }
    if (!apiKey.trim()) {
      setError('API key is required.');
      return;
    }
    if (!imageAsset?.uri) {
      setError('Pick an image first.');
      return;
    }

    setLoading(true);
    try {
      const form = new FormData();
      const name = imageAsset.fileName || `upload.${(imageAsset.uri.split('.').pop() || 'jpg').toLowerCase()}`;
      const type = imageAsset.mimeType || 'image/jpeg';

      form.append('file', {
        uri: imageAsset.uri,
        name,
        type,
      });

      const resp = await fetch(`${apiBaseUrl}/api/ai-detect`, {
        method: 'POST',
        headers: {
          'X-API-Key': apiKey.trim(),
        },
        body: form,
      });

      const data = await resp.json().catch(() => null);
      if (!resp.ok) {
        const msg = (data && (data.error || data.message)) || `Request failed (${resp.status})`;
        throw new Error(msg);
      }
      setResult(data);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Image Detection</Text>

      <Text style={styles.label}>API Base URL</Text>
      <Text style={styles.mono}>{apiBaseUrl || '(not set)'}</Text>

      <Text style={styles.label}>API Key</Text>
      <TextInput
        value={apiKey}
        onChangeText={setApiKey}
        autoCapitalize="none"
        autoCorrect={false}
        placeholder="mw_live_..."
        style={styles.input}
      />

      <View style={styles.row}>
        <Pressable style={styles.button} onPress={pickImage} disabled={loading}>
          <Text style={styles.buttonText}>Pick from Gallery</Text>
        </Pressable>
        <Pressable
          style={[styles.button, styles.primaryButton]}
          onPress={analyze}
          disabled={loading || !imageAsset}
        >
          <Text style={styles.buttonText}>{loading ? 'Analyzing…' : 'Analyze'}</Text>
        </Pressable>
      </View>

      {loading ? <ActivityIndicator style={styles.spacer} /> : null}

      {imageAsset?.uri ? (
        <Image source={{ uri: imageAsset.uri }} style={styles.preview} resizeMode="cover" />
      ) : null}

      {error ? <Text style={styles.error}>{error}</Text> : null}

      {result ? (
        <View style={styles.resultBox}>
          <Text style={styles.resultTitle}>Result</Text>
          <Text style={styles.resultText}>Prediction: {result.prediction}</Text>
          {result.confidence ? (
            <Text style={styles.resultText}>
              Confidence: real={Number(result.confidence.real).toFixed(3)} fake={Number(result.confidence.fake).toFixed(3)}
            </Text>
          ) : null}
        </View>
      ) : null}

      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#090',
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingTop: 64,
    paddingHorizontal: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 16,
  },
  label: {
    alignSelf: 'flex-start',
    color: 'rgba(255,255,255,0.9)',
    marginTop: 10,
    marginBottom: 6,
    fontWeight: '600',
  },
  mono: {
    alignSelf: 'flex-start',
    color: 'rgba(255,255,255,0.85)',
    fontFamily: 'Courier',
    marginBottom: 6,
  },
  input: {
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.95)',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
  },
  row: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 14,
    width: '100%',
  },
  button: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.25)',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  buttonText: {
    color: '#fff',
    fontWeight: '700',
  },
  spacer: {
    marginTop: 12,
  },
  preview: {
    width: '100%',
    height: 240,
    borderRadius: 14,
    marginTop: 14,
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  error: {
    width: '100%',
    color: '#ffebee',
    marginTop: 12,
    fontWeight: '700',
  },
  resultBox: {
    width: '100%',
    marginTop: 12,
    backgroundColor: 'rgba(0,0,0,0.25)',
    borderRadius: 14,
    padding: 12,
  },
  resultTitle: {
    color: '#fff',
    fontWeight: '800',
    marginBottom: 6,
  },
  resultText: {
    color: 'rgba(255,255,255,0.95)',
    fontWeight: '600',
  },
});
