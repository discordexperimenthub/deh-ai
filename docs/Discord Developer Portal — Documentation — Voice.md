# Discord Developer Portal — Documentation — Voice

## Voice Resource

## Voice State Object

Used to represent a user's voice connection status.

## Voice State Structure

| Field | Type | Description |
| --- | --- | --- |
| guild\_id? | snowflake | the guild id this voice state is for |
| channel\_id | ?snowflake | the channel id this user is connected to |
| user\_id | snowflake | the user id this voice state is for |
| member? | [guild member](https://ptb.discord.com/developers/docs/resources/guild#guild-member-object) object | the guild member this voice state is for |
| session\_id | string | the session id for this voice state |
| deaf | boolean | whether this user is deafened by the server |
| mute | boolean | whether this user is muted by the server |
| self\_deaf | boolean | whether this user is locally deafened |
| self\_mute | boolean | whether this user is locally muted |
| self\_stream? | boolean | whether this user is streaming using "Go Live" |
| self\_video | boolean | whether this user's camera is enabled |
| suppress | boolean | whether this user's permission to speak is denied |
| request\_to\_speak\_timestamp | ?ISO8601 timestamp | the time at which the user requested to speak |

## Example Voice State

```
{
  "channel_id": "157733188964188161",
  "user_id": "80351110224678912",
  "session_id": "90326bd25d71d39b9ef95b299e3872ff",
  "deaf": false,
  "mute": false,
  "self_deaf": false,
  "self_mute": true,
  "suppress": false,
  "request_to_speak_timestamp": "2021-03-31T18:45:31.297561+00:00"
}
```

## Voice Region Object

## Voice Region Structure

| Field | Type | Description |
| --- | --- | --- |
| id | string | unique ID for the region |
| name | string | name of the region |
| optimal | boolean | true for a single server that is closest to the current user's client |
| deprecated | boolean | whether this is a deprecated voice region (avoid switching to these) |
| custom | boolean | whether this is a custom voice region (used for events/etc) |

List Voice Regions[

](#list-voice-regions)
-------------------------------------------

GET/voice/regions

Returns an array of [voice region](https://ptb.discord.com/developers/docs/resources/voice#voice-region-object) objects that can be used when setting a voice or stage channel's [```rtc_region```](https://ptb.discord.com/developers/docs/resources/channel#channel-object-channel-structure).

## List Voice Regions

GET

/voice/regions

