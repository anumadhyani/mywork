document.addEventListener('DOMContentLoaded', function() {
  const fileInput = document.getElementById('fileInput');
  const analysisForm = document.getElementById('analysisForm');
  const analyzeButton = document.getElementById('analyzeButton');
  const loadingIndicator = document.getElementById('loadingIndicator');
  const resultsDiv = document.getElementById('results');
  const resultText = document.getElementById('resultText');
  const resultImages = document.getElementById('resultImages');
  const forensicOptions = document.getElementById('forensicOptions');
  const aiDetectRadio = document.getElementById('ai_detect');
  const forensicsRadio = document.getElementById('forensics');
  const signInButton = document.getElementById('signInButton');
  const signOutButton = document.getElementById('signOutButton');
  const authStatus = document.getElementById('authStatus');
  const authStatusSignedIn = document.getElementById('authStatusSignedIn');
  const signedOutView = document.getElementById('signedOutView');
  const signedInView = document.getElementById('signedInView');

  // Variable to store image data from context menu, if any
  let currentImageForAnalysis = null;
  let currentImageFileName = null;
  let currentImageSrcUrl = null; // To store original srcUrl from context menu

  const FLASK_BACKEND_URL = 'https://mywork-production.up.railway.app';

  // --- Google Sign-In + first-party JWT handling ---
  function _setAuthUi(jwtToken, email) {
    const isAuthed = !!jwtToken;

    // Signed-out: only show the Sign in button.
    if (signedOutView) signedOutView.classList.toggle('hidden', isAuthed);

    // Signed-in: show analyze UI + Sign out button.
    if (signedInView) signedInView.classList.toggle('hidden', !isAuthed);

    // Status copy is intentionally minimal; the visual state is the primary cue.
    const statusText = isAuthed ? (`Signed in${email ? ` as ${email}` : ''}.`) : 'Sign in to analyze images.';
    if (authStatus) authStatus.textContent = isAuthed ? '' : statusText;
    if (authStatusSignedIn) authStatusSignedIn.textContent = isAuthed ? statusText : '';

    // Reset analysis UI when switching to signed-out state.
    if (!isAuthed) {
      if (loadingIndicator) loadingIndicator.classList.add('hidden');
      if (resultsDiv) resultsDiv.classList.add('hidden');
      if (resultText) resultText.textContent = '';
      if (resultImages) resultImages.innerHTML = '';
    }
  }

  function _getStoredJwt() {
    // JWT is issued by our backend (/auth/google) and persisted in chrome.storage.sync.
    return new Promise((resolve) => {
      chrome.storage.sync.get(['authJwt', 'authEmail'], function(result) {
        const tok = (result && typeof result.authJwt === 'string') ? result.authJwt : '';
        const email = (result && typeof result.authEmail === 'string') ? result.authEmail : '';
        resolve({ token: tok || null, email: email || null });
      });
    });
  }

  async function _exchangeGoogleAccessTokenForJwt(googleAccessToken) {
    // Exchange Google token for our own JWT so the backend can auth requests consistently.
    const resp = await fetch(`${FLASK_BACKEND_URL}/auth/google`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ access_token: googleAccessToken })
    });

    if (!resp.ok) {
      let msg = `HTTP error! status: ${resp.status}`;
      try {
        const data = await resp.json();
        if (data && data.error) msg = data.error;
      } catch (e) {
      }
      throw new Error(msg);
    }
    return await resp.json();
  }

  async function _signInInteractive() {
    const getAuthToken = () => new Promise((resolve, reject) => {
      // chrome.identity uses the extension's oauth2 client_id/scopes from manifest.json.
      chrome.identity.getAuthToken({ interactive: true }, function(token) {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        resolve(token);
      });
    });

    const googleAccessToken = await getAuthToken();
    const jwtResp = await _exchangeGoogleAccessTokenForJwt(googleAccessToken);
    const jwtToken = (jwtResp && typeof jwtResp.access_token === 'string') ? jwtResp.access_token : null;
    if (!jwtToken) {
      throw new Error('invalid_jwt_response');
    }

    // Best-effort: fetch email for display only.
    const tiResp = await fetch(`https://oauth2.googleapis.com/tokeninfo?access_token=${encodeURIComponent(googleAccessToken)}`);
    let email = null;
    if (tiResp.ok) {
      try {
        const ti = await tiResp.json();
        if (ti && typeof ti.email === 'string') email = ti.email;
      } catch (e) {
      }
    }

    await new Promise((resolve) => {
      chrome.storage.sync.set({ authJwt: jwtToken, authEmail: email || '' }, resolve);
    });

    _setAuthUi(jwtToken, email);
    return jwtToken;
  }

  async function _signOut() {
    const removeAuthToken = () => new Promise((resolve) => {
      // Remove cached token so subsequent calls require interactive login again.
      chrome.identity.getAuthToken({ interactive: false }, function(token) {
        if (!token) {
          resolve(null);
          return;
        }
        chrome.identity.removeCachedAuthToken({ token }, function() {
          resolve(null);
        });
      });
    });

    await removeAuthToken();
    await new Promise((resolve) => chrome.storage.sync.remove(['authJwt', 'authEmail'], resolve));
    _setAuthUi(null, null);
  }

  if (signInButton) {
    signInButton.addEventListener('click', async function() {
      try {
        await _signInInteractive();
      } catch (e) {
        resultsDiv.classList.remove('hidden');
        resultText.textContent = `Sign-in failed: ${e.message}`;
      }
    });
  }

  if (signOutButton) {
    signOutButton.addEventListener('click', async function() {
      try {
        await _signOut();
      } catch (e) {
        resultsDiv.classList.remove('hidden');
        resultText.textContent = `Sign-out failed: ${e.message}`;
      }
    });
  }

  _getStoredJwt().then(({ token, email }) => {
    _setAuthUi(token, email);
  });

  // Toggle forensic options visibility based on radio button selection
  function toggleForensicOptions() {
    if (forensicsRadio.checked) {
      forensicOptions.classList.remove('hidden');
    } else {
      forensicOptions.classList.add('hidden');
    }
  }

  aiDetectRadio.addEventListener('change', toggleForensicOptions);
  forensicsRadio.addEventListener('change', toggleForensicOptions);
  toggleForensicOptions(); // Set initial state

  // --- Core Analysis Function (can be triggered by form submit or message from background.js) ---
  function _normalizeAiTag(prediction) {
    const p = (prediction || '').toString().toLowerCase();
    if (p.includes('ai') || p.includes('fake') || p.includes('synthetic')) return 'AI';
    if (p.includes('real')) return 'Real';
    return prediction || 'Unknown';
  }

  function _sentenceFromTag(tag) {
    if (tag === 'AI') return 'The image is likely AI-generated.';
    if (tag === 'Real') return 'The image is likely real.';
    return `The image result is: ${tag}.`;
  }

  // --- Result copy helpers (keep output consistent across model + C2PA enrichments) ---
  function _sentenceWithOrigin(baseSentence, origin) {
    // Preserve the model-derived sentence and append provenance if available.
    if (!origin) return baseSentence;
    return `${baseSentence} Source: ${origin}.`;
  }

  function _c2paOrigin(detailed) {
    if (!detailed) return null;

    try {
      if (typeof detailed === 'string') {
        const s = detailed.trim();
        if (!s) return null;
        return null;
      }

      if (typeof detailed !== 'object') return null;

      const manifestStore = detailed.manifest_store || detailed.manifestStore || null;
      const activeId = detailed.active_manifest || detailed.activeManifest || null;
      const active = (manifestStore && activeId && manifestStore[activeId]) ? manifestStore[activeId] : null;
      const claimGenerator = (active && (active.claim_generator || active.claimGenerator)) || detailed.claim_generator || detailed.claimGenerator || null;

      if (claimGenerator) return claimGenerator;
      return null;
    } catch (e) {
      return null;
    }
  }

  async function startAnalysis(fileOrBase64Data, fileName, analysisType, thresholdValue, minAreaThreshold) {
    console.log('DEBUG: startAnalysis called with:', {
        fileOrBase64Data: typeof fileOrBase64Data === 'string' ? fileOrBase64Data.substring(0, 50) + '...' : (fileOrBase64Data ? fileOrBase64Data.name : 'null'),
        fileName,
        analysisType,
        thresholdValue,
        minAreaThreshold
    });

    // Show loading indicator, hide results
    loadingIndicator.classList.remove('hidden');
    resultsDiv.classList.add('hidden');
    resultImages.innerHTML = ''; // Clear previous images
    resultText.textContent = ''; // Clear previous text

    let jwtToken = null;
    try {
      const stored = await _getStoredJwt();
      jwtToken = stored.token;
    } catch (e) {
    }
    if (!jwtToken) {
      loadingIndicator.classList.add('hidden');
      resultsDiv.classList.remove('hidden');
      resultText.textContent = 'Please sign in with Google.';
      return;
    }

    const abortController = new AbortController();
    let didTimeout = false;

    const slowTimer = setTimeout(() => {
      if (!didTimeout) {
        resultText.textContent = 'This is taking longer than expected.';
        resultsDiv.classList.remove('hidden');
      }
    }, 10000);

    const timeoutTimer = setTimeout(() => {
      didTimeout = true;
      abortController.abort();
      loadingIndicator.classList.add('hidden');
      resultsDiv.classList.remove('hidden');
      resultText.textContent = 'This is taking longer than usual. Please try again later.';
    }, 30000);

    let fileToUpload = null;
    let isBase64Input = false; // Flag to track if the input was base64 data
    let objectUrlToRevoke = null; // Variable to store object URL if created for manual file input

    if (fileOrBase64Data instanceof File) {
        fileToUpload = fileOrBase64Data;
        console.log('DEBUG: Input is a File object.');
    } else if (typeof fileOrBase64Data === 'string' && fileOrBase64Data.startsWith('data:image')) {
        isBase64Input = true; // Set the flag
        // Convert base64 to Blob then to File to append to FormData
        console.log('DEBUG: Input is base64 data. Converting to Blob/File...');
        const parts = fileOrBase64Data.split(';base64,');
        const contentType = parts[0].split(':')[1];
        const raw = window.atob(parts[1]);
        const rawLength = raw.length;
        const uInt8Array = new Uint8Array(rawLength);

        for (let i = 0; i < rawLength; ++i) {
            uInt8Array[i] = raw.charCodeAt(i);
        }
        const imageBlob = new Blob([uInt8Array], { type: contentType });
        fileToUpload = new File([imageBlob], fileName, { type: contentType });
        console.log('DEBUG: Base64 converted to File object. File name:', fileToUpload.name, 'type:', fileToUpload.type, 'size:', fileToUpload.size);
    } else {
        resultText.textContent = 'Error: No valid image data provided for analysis.';
        loadingIndicator.classList.add('hidden');
        resultsDiv.classList.remove('hidden');
        console.error('ERROR: Invalid image data type passed to startAnalysis.');
        return;
    }

    console.log('DEBUG: fileToUpload details: name=', fileToUpload.name, 'type=', fileToUpload.type, 'size=', fileToUpload.size);

    // Construct FormData
    const formData = new FormData();
    formData.append('file', fileToUpload);
    formData.append('analysis_type', analysisType);
    formData.append('threshold_value', thresholdValue);
    formData.append('min_area_threshold', minAreaThreshold);

    // Log FormData contents for debugging
    console.log('DEBUG: FormData content before sending:');
    for (let pair of formData.entries()) {
        console.log(pair[0]+ ', ' + pair[1]);
    }

    let apiEndpoint;
    if (analysisType === 'ai_detect') {
      apiEndpoint = `${FLASK_BACKEND_URL}/api/ai-detect`;
    } else if (analysisType === 'forensics') {
      resultText.textContent = 'Forensics is not enabled on the Railway backend yet.';
      loadingIndicator.classList.add('hidden');
      resultsDiv.classList.remove('hidden');
      return;
    } else {
      resultText.textContent = 'Error: Invalid analysis type selected.';
      loadingIndicator.classList.add('hidden');
      resultsDiv.classList.remove('hidden');
      console.error('ERROR: Invalid analysis type selected:', analysisType);
      return;
    }

    console.log('DEBUG: Sending POST request to:', apiEndpoint);
    try {
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
          // Use our first-party JWT instead of requiring end users to paste API keys.
          'Authorization': `Bearer ${jwtToken}`
        },
        body: formData,
        signal: abortController.signal
      });
      console.log('DEBUG: Received response from backend. Status:', response.status);

      if (didTimeout) {
        return;
      }

      if (!response.ok) {
        // Handle HTTP errors (e.g., 400, 500)
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('DEBUG: Backend response data:', data);

      if (didTimeout) {
        return;
      }

      if (data.status === 'success') {
        if (analysisType === 'ai_detect') {
          const tag = _normalizeAiTag(data.prediction);
          const baseSentence = _sentenceFromTag(tag);
          resultText.textContent = baseSentence;
          // Display original image thumbnail. If it's a manual file upload, create an object URL.
          // Store the object URL for later revocation.
          const imgSrc = isBase64Input ? fileOrBase64Data : (objectUrlToRevoke = URL.createObjectURL(fileToUpload));
          resultImages.innerHTML = `<img src="${imgSrc}" alt="Uploaded Image" style="max-width:200px; margin-top:10px;" />`;

          try {
            const c2paResponse = await fetch(`${FLASK_BACKEND_URL}/api/c2pa`, {
              method: 'POST',
              headers: {
                // Reuse JWT for all protected backend endpoints.
                'Authorization': `Bearer ${jwtToken}`
              },
              body: formData,
              signal: abortController.signal
            });
            if (c2paResponse.ok) {
              const c2paData = await c2paResponse.json();
              const origin = (c2paData && c2paData.status === 'success') ? _c2paOrigin(c2paData.c2pa) : null;
              if (origin) {
                resultText.textContent = _sentenceWithOrigin(baseSentence, origin);
              }
            }
          } catch (e) {
            console.warn('C2PA fetch failed:', e);
          }
        } else if (analysisType === 'forensics') {
          resultText.textContent = `Forensic Analysis: ${data.results.manipulation_reason}`;
          let imagesHtml = `
            <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
              <div style="text-align: center;">
                <h4>Original Image</h4>
                <img src="data:image/png;base64,${data.results.original_image_b64}" alt="Original Image" style="max-width:200px;" />
              </div>
          `;
          if (data.results.noise_overlay_b64) {
            imagesHtml += `
              <div style="text-align: center;">
                <h4>Noise Residual Overlay</h4>
                <img src="data:image/png;base64,${data.results.noise_overlay_b64}" alt="Noise Overlay" style="max-width:200px;" />
              </div>
            `;
          }
          if (data.results.freq_overlay_b64) {
            imagesHtml += `
              <div style="text-align: center;">
                <h4>Frequency Inconsistency Overlay</h4>
                <img src="data:image/png;base64,${data.results.freq_overlay_b64}" alt="Frequency Overlay" style="max-width:200px;" />
              </div>
            `;
          }
          if (data.results.ela_overlay_b64) {
            imagesHtml += `
              <div style="text-align: center;">
                <h4>ELA Overlay</h4>
                <img src="data:image/png;base64,${data.results.ela_overlay_b64}" alt="ELA Overlay" style="max-width:200px;" />
              </div>
            `;
          }
          imagesHtml += `</div>`;
          resultImages.innerHTML = imagesHtml;
        }
      } else {
        resultText.textContent = `Error: ${data.error || 'Unknown error from backend.'}`;
      }

    } catch (error) {
      console.error('Fetch error:', error);
      if (didTimeout || (error && error.name === 'AbortError')) {
        resultText.textContent = 'This is taking longer than usual. Please try again later.';
      } else if (error && error.message && error.message.toLowerCase().includes('rate limit exceeded')) {
        resultText.textContent = 'You have reached your daily limit. Please try buying more credits.';
      } else {
        resultText.textContent = `Analysis failed: ${error.message}. Please ensure the backend is running and accessible at ${FLASK_BACKEND_URL}.`;
      }
    } finally {
      clearTimeout(slowTimer);
      clearTimeout(timeoutTimer);
      loadingIndicator.classList.add('hidden');
      resultsDiv.classList.remove('hidden');
      // Revoke object URL if one was created
      if (objectUrlToRevoke) {
        URL.revokeObjectURL(objectUrlToRevoke);
        console.log('DEBUG: Object URL revoked.');
      }
    }
  }

  // Handle form submission for analysis
  analysisForm.addEventListener('submit', function(event) {
    event.preventDefault();
    console.log('DEBUG: Form submitted.');

    const analysisType = document.querySelector('input[name="analysis_type"]:checked').value;
    const thresholdValue = document.getElementById('threshold_value').value;
    const minAreaThreshold = document.getElementById('min_area_threshold').value;

    // Prioritize context menu image if available
    if (currentImageForAnalysis) {
        console.log('DEBUG: Using image from context menu for analysis.');
        startAnalysis(currentImageForAnalysis, currentImageFileName, analysisType, thresholdValue, minAreaThreshold);
    } else {
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select an image file or right-click an image on a webpage.');
            return;
        }
        console.log('DEBUG: Using image from manual file input for analysis.');
        startAnalysis(file, file.name, analysisType, thresholdValue, minAreaThreshold);
    }
  });

  // --- Retrieve data from chrome.storage.local when popup loads ---
  chrome.storage.local.get(['selectedImageSrcUrl', 'selectedImageData', 'selectedImageFileName', 'analysisError'], function(result) {
    console.log('DEBUG: popup.js trying to retrieve data from chrome.storage.local. Result:', result);

    if (result.analysisError) {
      resultsDiv.classList.remove('hidden');
      resultText.textContent = result.analysisError;
      chrome.storage.local.remove(['analysisError'], function() {
        if (chrome.runtime.lastError) {
          console.error('DEBUG: Error clearing analysisError:', chrome.runtime.lastError.message);
        }
      });
    }

    if (result.selectedImageData) {
      console.log('DEBUG: Image data found in storage. Populating UI.');
      currentImageForAnalysis = result.selectedImageData;
      currentImageFileName = result.selectedImageFileName;
      currentImageSrcUrl = result.selectedImageSrcUrl;

      // Display the selected image info in the popup UI
      fileInput.value = ''; // Clear manual file input
      fileInput.classList.add('hidden'); // Hide the manual file input
      fileInput.removeAttribute('required'); // REMOVE REQUIRED ATTRIBUTE WHEN HIDDEN

      // Remove any existing preview to avoid duplicates
      let existingPreview = document.querySelector('.image-preview');
      if (existingPreview) {
        existingPreview.remove();
      }

      // Create and display new preview for context-menu image
      const fileInputGroup = fileInput.closest('.input-group');
      const previewDiv = document.createElement('div');
      previewDiv.classList.add('image-preview');

      const previewImage = document.createElement('img');
      previewImage.src = currentImageForAnalysis;
      previewImage.style.maxWidth = '100px';
      previewImage.style.maxHeight = '100px';
      previewImage.style.marginTop = '10px';
      previewImage.style.display = 'block';
      previewDiv.appendChild(previewImage);

      const sourceText = document.createElement('p');
      sourceText.style.fontSize = '0.8em';
      sourceText.style.margin = '5px 0';
      sourceText.textContent = `Source: ${currentImageSrcUrl.substring(0, 40)}...`;
      previewDiv.appendChild(sourceText);

      fileInputGroup.appendChild(previewDiv);

      // Clear the data from storage to prevent re-processing on subsequent popup opens
      chrome.storage.local.remove(['selectedImageSrcUrl', 'selectedImageData', 'selectedImageFileName'], function() {
        if (chrome.runtime.lastError) {
          console.error('DEBUG: Error clearing storage:', chrome.runtime.lastError.message);
        }
      });

    } else {
      console.log('DEBUG: No image data found in storage.');
      fileInput.classList.remove('hidden'); // Ensure manual file input is visible
      fileInput.setAttribute('required', true); // RESTORE REQUIRED ATTRIBUTE WHEN VISIBLE
      let existingPreview = document.querySelector('.image-preview');
      if (existingPreview) {
          existingPreview.remove();
      }
    }
  });

  // Event listener for manual file input change (to reset if context menu image was present)
  fileInput.addEventListener('change', function() {
      console.log('DEBUG: Manual file input changed. Clearing context menu image data.');
      if (fileInput.files.length > 0) {
          currentImageForAnalysis = null; // Clear context menu image data
          currentImageFileName = null;
          currentImageSrcUrl = null;
          // Show the file input and remove preview
          fileInput.classList.remove('hidden');
          fileInput.setAttribute('required', true); // Ensure required is set if file input is used directly
          let existingPreview = document.querySelector('.image-preview');
          if (existingPreview) {
              existingPreview.remove();
          }
      }
  });

});