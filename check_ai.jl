using Images

using OCReract
using OpenAI

#= initialize OPENAI 

provider = OpenAI.OpenAIProvider(
    api_key=ENV["OPENAI_API_KEY"],
    base_url=ENV["OPENAI_BASE_URL_OVERRIDE"]
)

response = create_chat(
    provider,
    "gpt-3.5-turbo",
    [Dict("role" => "user", "content" => "Write some ancient Greek poetry")]
)

=#

secret_key = ENV["OPENAI_API_KEY"]
model = "gpt-3.5-turbo"


img_path = "/home/maarten/Documents/GIT/interesting_scripts/data/p04770487.jpg";
# With a disk file
run_tesseract(img_path, "./tmp/res.txt", psm=3, oem=1)
# Image in memory
img = load(img_path);
res_text = run_tesseract(img, psm=3, oem=1);
println(strip(res_text));

prompt = "Transform the following OCR output in DwC standard: " * strip(res_text)

r = create_chat(
    secret_key,
    model,
    [Dict("role" => "user", "content"=> prompt)]
  )

println(r.response[:choices][begin][:message][:content])