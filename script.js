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
