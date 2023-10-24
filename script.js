const audioFileInput = document.getElementById('audioFileInput');
const audioPlayer = document.getElementById('audioPlayer');
const waveformCanvas = document.getElementById('waveform');
const waveformContext = waveformCanvas.getContext('2d');

audioFileInput.addEventListener('change', handleFileUpload);

function handleFileUpload() {
    const file = audioFileInput.files[0];
    if (file) {
        // Set the audio source to the selected file
        audioPlayer.src = URL.createObjectURL(file);

        // Show the player panel
        audioPlayer.style.display = 'block';

        // Clear previous waveform
        waveformContext.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);

        // Create a Waveform.js instance for the canvas
        const waveform = new Waveform({
            container: waveformCanvas,
            interpolate: true,
        });
        
        // Load and render the waveform
        waveform.load(audioPlayer.src, () => {
            waveform.play();
        });
    }
}
document.getElementById("uploadButton").addEventListener("click", () => {
    const fileInput = document.getElementById("audioFileInput");
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append("audio", file);

        // Make an HTTP POST request to the API
        fetch("http://127.0.0.1:5000", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            alert("File uploaded and sent successfully!");
        })
        .catch(error => {
            console.error("Error: " + error);
            alert("An error occurred while uploading the file.");
        });
    } else {
        alert("Please select an audio file to upload.");
    }

});
