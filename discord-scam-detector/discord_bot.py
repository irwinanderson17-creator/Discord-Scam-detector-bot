import os
import discord
import joblib

from discord import app_commands
from dotenv import load_dotenv

# Load the Discord token
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Load the trained scam detector
model = joblib.load("scam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class ScamBot(discord.Client):

    def __init__(self):
        intents = discord.Intents.default()

        super().__init__(intents=intents)

        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()


bot = ScamBot()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.tree.command(
    name="checkmessage",
    description="Check if a Discord message may be a scam"
)
async def checkmessage(interaction: discord.Interaction, message: str):

    # Turn the message into numbers
    message_tfidf = vectorizer.transform([message])

    # Make a prediction
    prediction = model.predict(message_tfidf)[0]

    # Get scam probability
    scam_probability = model.predict_proba(message_tfidf)[0][1]
    probability_percent = round(scam_probability * 100, 2)

    if prediction == 1:
        result = "⚠️ Possible Scam/Phishing"
    else:
        result = "✅ Likely Normal"

    await interaction.response.send_message(
        f"**Prediction:** {result}\n"
        f"**Scam probability:** {probability_percent}%"
    )

bot.run(TOKEN)



