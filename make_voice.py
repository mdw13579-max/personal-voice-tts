from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

script = """
Alright… here’s the honest truth.

After weeks of wrestling with the same problem,
I was this close to throwing in the towel.

I stayed late.
Tried new angles.
Even reached out for help.
And still… nothing clicked.

But deep down,
something in me wasn’t ready to quit just yet.

So instead of giving up,
I stepped back.
Took a real break.
Cleared my head.

And funny enough—
that turned out to be the smartest move of all.

Sometimes the win
isn’t pushing harder.

It’s knowing when to pause…
so you can come back stronger.

You got this.
"""

audio = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input=(
        "Upbeat American middle-aged coach voice. "
        "Warm, confident, lightly humorous, clear pronunciation.\n\n"
        + script
    ),
)

with open("coach_voice.mp3", "wb") as f:
    if hasattr(audio, "read"):
        f.write(audio.read())
    else:
        f.write(audio.content)

print("MP3 created → coach_voice.mp3")
