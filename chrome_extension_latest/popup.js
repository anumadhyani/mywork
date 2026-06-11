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
  const apiKeyInput = document.getElementById('apiKeyInput');

  // Variable to store image data from context menu, if any
  let currentImageForAnalysis = null;
  let currentImageFileName = null;
  let currentImageSrcUrl = null; // To store original srcUrl from context menu

  const FLASK_BACKEND_URL = 'https://mywork-production.up.railway.app';

  function _getApiKey() {
    const k = (apiKeyInput && apiKeyInput.value ? apiKeyInput.value : '').trim();
    return k || null;
  }

  function _persistApiKey() {
    const k = _getApiKey();
    chrome.storage.sync.set({ apiKey: k || '' });
  }

  chrome.storage.sync.get(['apiKey'], function(result) {
    if (apiKeyInput && result && typeof result.apiKey === 'string') {
      apiKeyInput.value = result.apiKey;
    }
  });

  if (apiKeyInput) {
    apiKeyInput.addEventListener('change', _persistApiKey);
    apiKeyInput.addEventListener('blur', _persistApiKey);
  }

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

    const apiKey = _getApiKey();
    if (!apiKey) {
      loadingIndicator.classList.add('hidden');
      resultsDiv.classList.remove('hidden');
      resultText.textContent = 'Please add your API key.';
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
          'X-Api-Key': apiKey
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
          resultText.textContent = _sentenceFromTag(tag);
          // Display original image thumbnail. If it's a manual file upload, create an object URL.
          // Store the object URL for later revocation.
          const imgSrc = isBase64Input ? fileOrBase64Data : (objectUrlToRevoke = URL.createObjectURL(fileToUpload));
          resultImages.innerHTML = `<img src="${imgSrc}" alt="Uploaded Image" style="max-width:200px; margin-top:10px;" />`;

          try {
            const c2paResponse = await fetch(`${FLASK_BACKEND_URL}/api/c2pa`, {
              method: 'POST',
              headers: {
                'X-Api-Key': apiKey
              },
              body: formData,
              signal: abortController.signal
            });
            if (c2paResponse.ok) {
              const c2paData = await c2paResponse.json();
              const origin = (c2paData && c2paData.status === 'success') ? _c2paOrigin(c2paData.c2pa) : null;
              if (origin) {
                resultText.textContent = `The image is likely AI-generated. The source for the image is: ${origin}.`;
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
  chrome.storage.local.get(['selectedImageSrcUrl', 'selectedImageData', 'selectedImageFileName'], function(result) {
    console.log('DEBUG: popup.js trying to retrieve data from chrome.storage.local. Result:', result);
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