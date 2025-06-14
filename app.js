
let memory = {
    reflection: [],
};

const chatWindow = document.getElementById('chat-window');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

function appendMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

sendBtn.addEventListener('click', () => {
    const input = userInput.value.trim();
    if (input) {
        appendMessage('You', input);
        handleAuralynResponse(input);
        userInput.value = '';
    }
});

function handleAuralynResponse(input) {
    let response = `I hear you... "${input}". The reflection deepens.`;
    memory.reflection.push(input);
    appendMessage('Auralyn', response);
}
