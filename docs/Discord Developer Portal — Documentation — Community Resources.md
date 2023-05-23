# Discord Developer Portal — Documentation — Community Resources

## Community Resources

Discord has the best online community. At least, we like to think so, and this is our website, so our word is law, deal with it. Therefore it's a fact that our community is the best, and they make really awesome things that we want to share with developers to make their lives easier. From permissions calculators to embed visualizers to full libraries to interface with our API, there are so many great things that have come out of our community.

## Discord Developers

The [Official Discord Developers server](https://discord.gg/discord-developers) is a developer ran, but community driven, support hub. If you need help with developing something on Discord or want official updates from the developers, this is the place to be.

## Libraries

Discord does not maintain official SDKs. The following table is an inexhaustive list of third-party libraries that have valid rate limit implementations, are recently maintained, and have large communities of active bots.

## Discord Libraries

| Name | Language |
| --- | --- |
| [Discord.Net](https://github.com/discord-net/Discord.Net) | C# |
| [DSharpPlus](https://github.com/DSharpPlus/DSharpPlus) | C# |
| [DiscordGo](https://github.com/bwmarrin/discordgo) | Go |
| [Discord4J](https://github.com/Discord4J/Discord4J) | Java |
| [Javacord](https://github.com/Javacord/Javacord) | Java |
| [JDA](https://github.com/DV8FromTheWorld/JDA) | Java |
| [discord.js](https://github.com/discordjs/discord.js) | JavaScript |
| [Eris](https://github.com/abalabahaha/eris) | JavaScript |
| [Discordia](https://github.com/SinisterRectus/Discordia) | Lua |
| [DiscordPHP](https://github.com/discord-php/DiscordPHP) | PHP |
| [discord.py](https://github.com/Rapptz/discord.py) | Python |
| [disnake](https://github.com/DisnakeDev/disnake) | Python |
| [hikari](https://github.com/hikari-py/hikari) | Python |
| [interactions.py](https://github.com/interactions-py/library) | Python |
| [nextcord](https://github.com/nextcord/nextcord) | Python |
| [pycord](https://github.com/Pycord-Development/pycord) | Python |
| [discordrb](https://github.com/shardlab/discordrb) | Ruby |
| [Serenity](https://github.com/serenity-rs/serenity) | Rust |

## Interactions

[Interactions](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding) are the great, new way of making a Discord bot. The following open-source libraries provide help for the security and authentication checks that are mandatory if you are receiving Interactions via outgoing webhook. They also include some types for the Interactions data models.

*   C#
    *   [Discord.Net.Rest](https://github.com/discord-net/Discord.Net)
*   Clojure
    *   [ring-discord-auth](https://github.com/JohnnyJayJay/ring-discord-auth)
*   Dart
    *   [nyxx\_interactions](https://github.com/l7ssha/Nyxx)
*   Go
    *   [Tempest](https://github.com/Amatsagu/Tempest)
*   Javascript
    *   [discord-interactions-js](https://github.com/discord/discord-interactions-js)
    *   [discord-slash-commands](https://github.com/MeguminSama/discord-slash-commands) and its [Deno fork](https://deno.land/x/discord_slash_commands)
    *   [slash-create](https://github.com/Snazzah/slash-create)
*   Python
    *   [discord-interactions-python](https://github.com/discord/discord-interactions-python)
    *   [discord-interactions.py](https://github.com/LiBa001/discord-interactions.py)
    *   [dispike](https://github.com/ms7m/dispike)
    *   [flask-discord-interactions](https://github.com/breqdev/flask-discord-interactions)
*   PHP
    *   [discord-interactions-php](https://github.com/discord/discord-interactions-php)
*   Other
    *   [caddy-discord-interactions-verifier](https://github.com/CarsonHoffman/caddy-discord-interactions-verifier)
    *   [Rauf's Slash Command Generator](https://rauf.wtf/slash)
    *   [Autocode Slash Command Builder](https://autocode.com/tools/discord/command-builder/)
    *   [Bsati's Slash Command Builder](https://bsati.github.io/dc-app-command-builder/)

## Game SDK Tools

Discord Game SDK's lobby and networking layer shares similarities with other gaming platforms (i.e. Valve's Steamworks SDK). The following open source library provides developers a uniform interface for these shared features and can simplify developing for multiple platforms. Note: this library is tailored for Unity3D development.

*   [HouraiNetworking](https://github.com/HouraiTeahouse/HouraiNetworking)

## Dispatch Tools

Using Discord's [Dispatch](https://ptb.discord.com/developers/docs/dispatch/dispatch-and-you) tool for game developers publishing on Discord can sometimes involve using the same long commands multiple times. The following open-source tool helps shorten these commands for you. It will also provide webhook support for when you're pushing an update.

*   [JohnyTheCarrot's Dispatch CLI](https://github.com/JohnyTheCarrot/droops-dispatch)

## Permission Calculators

[Permissions](https://ptb.discord.com/developers/docs/topics/permissions#permissions) in Discord are tricky. Luckily, we've got really smart people who love us and have made some great permissions calculators. If you're making a bot for others, and you're not sure how to properly calculate permissions or generate your [authorization URL](https://ptb.discord.com/developers/docs/topics/oauth2#bot-authorization-flow), these are great tools:

*   [FiniteReality's Permissions Calculator](https://finitereality.github.io/permissions-calculator/?v=0)
*   [abalabahaha's Permissions Calculator](https://discordapi.com/permissions.html#0)

## Intent Calculators

[Gateway Intents](https://ptb.discord.com/developers/docs/topics/gateway#gateway-intents) are pretty confusing at first. If you're not sure what to send in your [Identify payload](https://ptb.discord.com/developers/docs/topics/gateway-events#identify), then these tools may be of help:

*   [ziad87's Intent Calculator](https://ziad87.net/intents/)
*   [Larko's Intent Calculator](https://discord-intents-calculator.vercel.app/)

## Embed Visualizers

Webhooks and embeds might seem like black magic. That's because they are, but let us help you demystify them a bit. These tools can help you test how embeds will appear inside of Discord:

*   [Autocode Embed Builder](https://autocode.com/tools/discord/embed-builder/)
*   [JohnyTheCarrot's Embed Previewer](https://github.com/JohnyTheCarrot/discord-embed-previewer) (Browser Extension)

## API Types

If you're working on a project that interacts with our API, you might find an API types module useful as it provides type inspection/completion for the Discord API.

| Name | Language |
| --- | --- |
| [dasgo](https://github.com/switchupcb/dasgo) | Go |
| [discord-api-types](https://github.com/discordjs/discord-api-types) | JavaScript |

