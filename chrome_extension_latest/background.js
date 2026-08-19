const FLASK_BACKEND_URL = 'https://mywork-production.up.railway.app';

chrome.runtime.onInstalled.addListener(() => {
  console.log('background.js: Extension installed. Creating context menu.');
  chrome.contextMenus.create({
    id: 'analyzeImageContextMenu',
    title: 'Analyze Image with AI Tool',
    contexts: ['image']
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  console.log('background.js: Context menu clicked. Info:', info);
  if (info.menuItemId === 'analyzeImageContextMenu' && info.mediaType === 'image') {
    const imageUrl = info.srcUrl;
    console.log('background.js: Image URL selected:', imageUrl);

    try {
      console.log('background.js: Attempting to fetch image from URL:', imageUrl);
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      console.log('background.js: Image fetched successfully. Converting to Blob.');
      const imageBlob = await response.blob();

      const reader = new FileReader();
      reader.readAsDataURL(imageBlob);
      reader.onloadend = () => {
        const base64data = reader.result;
        // Start with the part after the last slash
        let fileName = imageUrl.substring(imageUrl.lastIndexOf('/') + 1);

        // Strip query parameters from the filename, if any
        const queryParamIndex = fileName.indexOf('?');
        if (queryParamIndex > -1) {
            fileName = fileName.substring(0, queryParamIndex);
        }

        // Ensure fileName has an extension. If not, try to derive from blob type or default.
        if (!fileName.includes('.') && imageBlob.type) {
            const extension = imageBlob.type.split('/')[1];
            if (extension) {
                fileName = `${fileName}.${extension.replace('jpeg', 'jpg')}`; // Standardize jpeg to jpg
            } else {
                fileName = `${fileName}.png`; // Default fallback
            }
        } else if (!fileName.includes('.')) { // If no extension and no blob type, fallback
            fileName = `${fileName}.png`;
        }
        // Ensure a name is always present, even if it was just a domain or empty after substring
        fileName = fileName || 'selected_image.png';

        console.log('background.js: Image converted to Base64. Storing in chrome.storage.local. File name:', fileName);

        chrome.storage.local.set({
          'selectedImageSrcUrl': imageUrl,
          'selectedImageData': base64data,
          'selectedImageFileName': fileName
        }, () => {
          if (chrome.runtime.lastError) {
            console.error('background.js: Error storing data:', chrome.runtime.lastError.message);
          } else {
            console.log('background.js: Data stored in chrome.storage.local. Opening popup.');
            chrome.action.openPopup(); // Open the extension popup
          }
        });
      };
      reader.onerror = () => {
        console.error('background.js: FileReader error during Base64 conversion:', reader.error);
        // Error handling if needed
      };

    } catch (error) {
      console.error('background.js: Error fetching or processing image:', error);
      // Error handling if needed
    }
  }
});

console.log('background.js service worker loaded with enhanced functionality.');