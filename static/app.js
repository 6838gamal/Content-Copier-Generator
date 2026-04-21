document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const topic = document.getElementById("topic").value;
    const length = document.getElementById("length").value;

    const res = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ topic, length })
    });

    const data = await res.json();
    document.getElementById("result").innerText = data.generated;
});

document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = document.getElementById("file").files[0];
    const formData = new FormData();
    formData.append("file", file);

    await fetch("/upload", {
        method: "POST",
        body: formData
    });

    alert("تم الرفع بنجاح");
});
