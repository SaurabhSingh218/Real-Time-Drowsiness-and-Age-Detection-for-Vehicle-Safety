const videoFeed = document.getElementById("videoFeed");
const exitScreen = document.getElementById("exitScreen");

// Open Webcam
document.getElementById("webcamBtn").addEventListener("click", () => {
    videoFeed.src = "/video_feed?source=webcam";
    exitScreen.style.display = "none";
});

// Upload Video
document.getElementById("videoInput").addEventListener("change", function() {
    if (!this.files.length) return;
    let formData = new FormData();
    formData.append("file", this.files[0]);

    fetch("/upload_video", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
            videoFeed.src = "/video_feed?source=video&file=" + data.file_path;
            exitScreen.style.display = "none";
        });
});

// Upload Image
document.getElementById("imageInput").addEventListener("change", function() {
    if (!this.files.length) return;
    let formData = new FormData();
    formData.append("file", this.files[0]);

    fetch("/upload_image", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
            videoFeed.src = "/video_feed?source=image&file=" + data.file_path;
            exitScreen.style.display = "none";
        });
});

// Exit Button
document.getElementById("exitBtn").addEventListener("click", () => {
    videoFeed.src = "";
    exitScreen.style.display = "flex";
});
