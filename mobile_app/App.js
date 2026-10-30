import { StatusBar } from 'expo-status-bar';
import Constants from 'expo-constants';
import * as ImagePicker from 'expo-image-picker';
import * as SecureStore from 'expo-secure-store';
import { useEffect, useMemo, useState } from 'react';
import {
  GoogleOneTapSignIn,
  isSuccessResponse,
  isCancelledResponse,
} from 'react-native-nitro-google-signin';
import {
  ActivityIndicator,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

const JWT_KEY = 'truthlens.jwt';
const EMAIL_KEY = 'truthlens.email';

// Same wording as the Chrome extension for non-AI outcomes.
const NON_AI_SENTENCE = 'No strong AI signals detected.';

function tagFromBand(band) {
  const b = (band || '').toString().toLowerCase();
  if (b === 'ai' || b === 'likely_ai') return 'AI';
  if (b === 'real' || b === 'likely_real') return 'Real';
  if (b === 'uncertain') return 'Uncertain';
  return 'Unknown';
}

function displaySentence(result) {
  const band = (result?.band || '').toString().toLowerCase();
  const nonAi = band === 'uncertain' || band === 'likely_real' || band === 'real';
  if (nonAi) return NON_AI_SENTENCE;
  if (result?.prediction) return String(result.prediction);
  const tag = tagFromBand(band);
  if (tag === 'AI') return 'This image is likely AI-generated.';
  if (tag === 'Real') return NON_AI_SENTENCE;
  return 'Analysis complete.';
}

function bandColors(band) {
  const b = (band || '').toString().toLowerCase();
  if (b === 'ai' || b === 'likely_ai') return { bg: '#7f1d1d', fg: '#fecaca' };
  if (b === 'real' || b === 'likely_real') return { bg: '#14532d', fg: '#bbf7d0' };
  return { bg: '#713f12', fg: '#fde68a' };
}

export default function App() {
  const { apiBaseUrl, webClientId } = useMemo(() => {
    const extra = Constants?.expoConfig?.extra || {};
    const url = (extra.apiBaseUrl || extra.API_BASE_URL || '').toString().trim().replace(/\/$/, '');
    const wcid = (extra.googleWebClientId || '').toString().trim();
    return { apiBaseUrl: url, webClientId: wcid };
  }, []);

  const [jwt, setJwt] = useState(null);
  const [email, setEmail] = useState('');
  const [signingIn, setSigningIn] = useState(false);
  const [booting, setBooting] = useState(true);
  const [imageAsset, setImageAsset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [c2pa, setC2pa] = useState(null);

  // Configure Google Sign-In once and restore any saved session.
  useEffect(() => {
    GoogleOneTapSignIn.configure({
      webClientId: webClientId || undefined,
      scopes: ['openid', 'email', 'profile'],
    });

    (async () => {
      try {
        const [savedJwt, savedEmail] = await Promise.all([
          SecureStore.getItemAsync(JWT_KEY),
          SecureStore.getItemAsync(EMAIL_KEY),
        ]);
        if (savedJwt) setJwt(savedJwt);
        if (savedEmail) setEmail(savedEmail);
      } catch {
        // SecureStore unavailable; user can sign in again.
      } finally {
        setBooting(false);
      }
    })();
  }, [webClientId]);

  const signIn = async () => {
    setError('');
    setSigningIn(true);
    try {
      if (Platform.OS === 'android') {
        try {
          await GoogleOneTapSignIn.checkPlayServices(false);
        } catch {
          // Keep going; sign-in may still work on some devices.
        }
      }

      // Explicit Google sign-in UI (matches chrome.identity interactive flow).
      const res = await GoogleOneTapSignIn.presentExplicitSignIn();
      if (!isSuccessResponse(res)) {
        if (!isCancelledResponse(res)) setError('Google sign-in did not return a credential.');
        return;
      }

      const idToken = res.data?.idToken || res.data?.user?.idToken;
      if (!idToken) {
        setError('No Google ID token returned. Check the Web client ID configuration.');
        return;
      }

      // Exchange the Google ID token for our first-party JWT (same as the extension).
      const resp = await fetch(`${apiBaseUrl}/auth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id_token: idToken }),
      });
      const data = await resp.json().catch(() => null);
      if (!resp.ok || !data?.access_token) {
        throw new Error((data && data.error) || `Auth exchange failed (${resp.status})`);
      }

      const userEmail = res.data?.user?.email || '';
      setJwt(data.access_token);
      setEmail(userEmail);
      await SecureStore.setItemAsync(JWT_KEY, data.access_token);
      if (userEmail) await SecureStore.setItemAsync(EMAIL_KEY, userEmail);
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setSigningIn(false);
    }
  };

  const signOut = async () => {
    setError('');
    setResult(null);
    setC2pa(null);
    setJwt(null);
    setEmail('');
    try {
      await GoogleOneTapSignIn.signOut();
    } catch {
      // Non-fatal.
    }
    try {
      await SecureStore.deleteItemAsync(JWT_KEY);
      await SecureStore.deleteItemAsync(EMAIL_KEY);
    } catch {
      // Non-fatal.
    }
  };

  const pickImage = async () => {
    setError('');
    setResult(null);
    setC2pa(null);

    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      setError('Gallery permission is required.');
      return;
    }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });

    if (res.canceled) return;
    setImageAsset(res.assets?.[0] || null);
  };

  const buildForm = () => {
    const form = new FormData();
    const name =
      imageAsset.fileName ||
      `upload.${(imageAsset.uri.split('.').pop() || 'jpg').toLowerCase()}`;
    const type = imageAsset.mimeType || 'image/jpeg';
    form.append('file', { uri: imageAsset.uri, name, type });
    return form;
  };

  const analyze = async () => {
    setError('');
    setResult(null);
    setC2pa(null);

    if (!apiBaseUrl) {
      setError('Missing apiBaseUrl. Set it in mobile_app/app.json (expo.extra.apiBaseUrl).');
      return;
    }
    if (!jwt) {
      setError('Sign in first.');
      return;
    }
    if (!imageAsset?.uri) {
      setError('Pick an image first.');
      return;
    }

    setLoading(true);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/ai-detect`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${jwt}`,
          'X-Client-Channel': 'mobile_app',
        },
        body: buildForm(),
      });

      const data = await resp.json().catch(() => null);
      if (!resp.ok) {
        if (resp.status === 401) {
          setJwt(null);
          await SecureStore.deleteItemAsync(JWT_KEY).catch(() => {});
          throw new Error('Session expired. Please sign in again.');
        }
        const msg = (data && (data.error || data.message)) || `Request failed (${resp.status})`;
        throw new Error(msg);
      }
      setResult(data);

      // C2PA provenance check (same as the extension runs alongside detection).
      try {
        const c2paResp = await fetch(`${apiBaseUrl}/api/c2pa`, {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${jwt}`,
            'X-Client-Channel': 'mobile_app',
          },
          body: buildForm(),
        });
        const c2paData = await c2paResp.json().catch(() => null);
        if (c2paResp.ok && c2paData) setC2pa(c2paData);
      } catch {
        // C2PA is best-effort; ignore failures.
      }
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  };

  if (booting) {
    return (
      <View style={[styles.container, styles.center]}>
        <ActivityIndicator color="#fff" />
        <StatusBar style="light" />
      </View>
    );
  }

  if (!jwt) {
    return (
      <View style={[styles.container, styles.center]}>
        <Text style={styles.title}>TruthLens</Text>
        <Text style={styles.subtitle}>Analyze images for AI generation.</Text>
        <Pressable
          style={[styles.button, styles.primaryButton, styles.signInButton]}
          onPress={signIn}
          disabled={signingIn || !webClientId}
        >
          <Text style={styles.buttonText}>
            {signingIn ? 'Signing in…' : 'Sign in with Google'}
          </Text>
        </Pressable>
        {!webClientId ? (
          <Text style={styles.error}>
            Missing googleWebClientId in app.json (expo.extra.googleWebClientId).
          </Text>
        ) : null}
        {error ? <Text style={styles.error}>{error}</Text> : null}
        <StatusBar style="light" />
      </View>
    );
  }

  const band = (result?.band || '').toString().toLowerCase();
  const colors = bandColors(band);

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
      <View style={styles.headerRow}>
        <View>
          <Text style={styles.title}>TruthLens</Text>
          {email ? <Text style={styles.email}>{email}</Text> : null}
        </View>
        <Pressable onPress={signOut} style={styles.signOutButton}>
          <Text style={styles.signOutText}>Sign out</Text>
        </Pressable>
      </View>

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

      {loading ? <ActivityIndicator style={styles.spacer} color="#fff" /> : null}

      {imageAsset?.uri ? (
        <Image source={{ uri: imageAsset.uri }} style={styles.preview} resizeMode="cover" />
      ) : null}

      {error ? <Text style={styles.error}>{error}</Text> : null}

      {result ? (
        <View style={styles.resultBox}>
          <View style={[styles.badge, { backgroundColor: colors.bg }]}>
            <Text style={[styles.badgeText, { color: colors.fg }]}>
              {tagFromBand(band)}
            </Text>
          </View>
          <Text style={styles.resultText}>{displaySentence(result)}</Text>
          {result.confidence ? (
            <Text style={styles.resultSubtext}>
              Confidence: real={Number(result.confidence.real).toFixed(3)}{' '}
              fake={Number(result.confidence.fake).toFixed(3)}
            </Text>
          ) : null}
          {result.backend ? (
            <Text style={styles.resultSubtext}>Backend: {result.backend}</Text>
          ) : null}
        </View>
      ) : null}

      {c2pa ? (
        <View style={styles.resultBox}>
          <Text style={styles.resultTitle}>Provenance (C2PA)</Text>
          <Text style={styles.resultSubtext}>
            {c2pa.status === 'success'
              ? 'Content credentials found.'
              : c2pa.error || 'No content credentials found.'}
          </Text>
        </View>
      ) : null}

      <StatusBar style="light" />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  center: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  scroll: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollContent: {
    paddingTop: 64,
    paddingHorizontal: 16,
    paddingBottom: 40,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: '#fff',
  },
  subtitle: {
    color: 'rgba(255,255,255,0.75)',
    marginTop: 6,
    marginBottom: 24,
  },
  email: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    marginTop: 4,
  },
  signOutButton: {
    paddingVertical: 6,
    paddingHorizontal: 10,
  },
  signOutText: {
    color: '#93c5fd',
    fontWeight: '600',
  },
  signInButton: {
    width: '100%',
    flex: 0,
    marginTop: 8,
  },
  row: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 14,
    width: '100%',
  },
  button: {
    flex: 1,
    backgroundColor: '#1e293b',
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: 'center',
  },
  primaryButton: {
    backgroundColor: '#2563eb',
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
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  error: {
    width: '100%',
    color: '#fca5a5',
    marginTop: 12,
    fontWeight: '700',
    textAlign: 'center',
  },
  resultBox: {
    width: '100%',
    marginTop: 12,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderRadius: 14,
    padding: 14,
  },
  badge: {
    alignSelf: 'flex-start',
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 4,
    marginBottom: 8,
  },
  badgeText: {
    fontWeight: '800',
    fontSize: 12,
    letterSpacing: 0.5,
  },
  resultTitle: {
    color: '#fff',
    fontWeight: '800',
    marginBottom: 6,
  },
  resultText: {
    color: 'rgba(255,255,255,0.95)',
    fontWeight: '700',
    fontSize: 16,
    marginBottom: 6,
  },
  resultSubtext: {
    color: 'rgba(255,255,255,0.75)',
    fontWeight: '600',
    fontSize: 13,
  },
});
