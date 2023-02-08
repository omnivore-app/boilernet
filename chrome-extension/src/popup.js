const highlightContentButton = document.getElementById('highlightContent');

highlightContentButton.onclick = async function(_element) {
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    await chrome.tabs.sendMessage(tab.id, {text: "sendDocumentRepresentation"});
    // disable button
    highlightContentButton.textContent = "Working...";
    highlightContentButton.disabled = true;
};

chrome.runtime.onMessage.addListener(function (msg, _sender, _sendResponse) {
    if (msg.text === 'enableButton') {
        highlightContentButton.textContent = "Highlight content";
        highlightContentButton.disabled = false;
    }
});
