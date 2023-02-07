const highlightContentButton = document.getElementById('highlightContent');

highlightContentButton.onclick = function(_element) {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        console.log('sending message to tab: ', tabs[0].id)
        chrome.tabs.sendMessage(tabs[0].id, {text: "sendDocumentRepresentation"}, function(response) {
            console.log('response: ', response);
            // disable button
            highlightContentButton.textContent = "Working...";
            highlightContentButton.disabled = true;
        });
    });
};

chrome.runtime.onMessage.addListener(function (msg, sender, _sendResponse) {
    console.log('popup.js received msg: ' + msg.text);
    if (msg.text === 'enableButton') {
        highlightContentButton.textContent = "Highlight content";
        highlightContentButton.disabled = false;
    }
});
