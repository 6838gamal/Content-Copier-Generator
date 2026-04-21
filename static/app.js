document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const topic = document.getElementById("topic").value;

    const res = await fetch("/generate", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ topic })
    });

    const data = await res.json();

    document.getElementById("result").innerText = data.generated;
});
