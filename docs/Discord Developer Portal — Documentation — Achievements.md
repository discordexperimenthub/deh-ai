# Discord Developer Portal — Documentation — Achievements

## Achievements

Need help with the SDK? Talk to us in the [Discord Developers Server](https://discord.gg/discord-developers)!

Selling SKUs on Discord has now been discontinued as of March 1, 2022. [Read here for more info.](https://support-dev.discord.com/hc/en-us/articles/6309018858647-Self-serve-Game-Selling-Deprecation)

The GameSDK's Achievements, Applications, Voice, Images, Lobbies, Networking, Storage, and Store (purchases and discounts) features have been deprecated, and will be decommissioned on May 2, 2023. [Read more](https://ptb.discord.com/developers/docs/change-log#gamesdk-feature-deprecation)

There's no feeling quite like accomplishing a goal that you've set out to achieve. Is killing 1000 zombies in a game as great an achievement as climbing Mt. Everest? Of course it is, and I didn't even have to leave my house. So get off my back, society.

Anyway—Discord has achievements! Show your players just how successful they are.

Achievements are managed in the [Developer Portal](https://discord.comhttps://ptb.discord.com/developers/applications). Head over to your application --> ```Achievements``` to create and manage achievements for your game. You'll give them an icon, a name, and a description; then they'll be assigned an id.

You can also mark achievements as ```secret``` and ```secure```. "Secret" achievements will not be shown to the user until they've unlocked them. "Secure" achievements can only be set via HTTP calls from your server, not by a game client using the SDK. No cheaters here!

## Data Models

## Achievement Struct

| name | type | description |
| --- | --- | --- |
| application\_id | Int64 | Unique ID of the application |
| name | string | Name of the achievement |
| name\_localizations | ?dictionary with keys as [available locales](https://ptb.discord.com/developers/docs/reference#locales) | Localization dictionary for the ```name``` field |
| description | string | Description of the achievement |
| description\_localizations | ?dictionary with keys as [available locales](https://ptb.discord.com/developers/docs/reference#locales) | Localization dictionary for the ```description``` field |
| secret | boolean | If the achievement is secret |
| secure | boolean | If the achievement is secure |
| id | Int64 | Unique ID of the achievement |
| icon\_hash | string | [Hash of the icon](https://ptb.discord.com/developers/docs/reference#image-formatting) |

## User Achievement Struct

| name | type | description |
| --- | --- | --- |
| UserId | Int64 | the unique ID of the user working on the achievement |
| AchievementId | Int64 | the unique ID of the achievement |
| PercentComplete | UInt8 | how far along the user is to completing the achievement (0-100) |
| UnlockedAt | string | the timestamp at which the user completed the achievement (PercentComplete was set to 100) |

## SetUserAchievement

Updates the current user's status for a given achievement. If ```percentComplete``` is set to ```100```, the ```UnlockedAt``` field will be automatically updated with the current timestamp.

Returns ```Discord.Result``` via callback.

## Parameters

| name | type | description |
| --- | --- | --- |
| achievementId | Int64 | the ID of the achievement to update |
| percentComplete | UInt8 | the user's updated percentage progress |

## Example

```
achievementManager.SetUserAchievement(580159119969878046, 25, (res) =>
{
  if (res == Discord.Result.Ok)
  {
    Console.WriteLine("Achievement updated for user");
  }
});
```

## FetchUserAchievements

Loads a stable list of the current user's achievements to iterate over. If the user has any achievements, do your iteration within the callback of this function.

Returns ```Discord.Result``` via callback.

Remember to only iterate when there are results!

## Parameters

None.

## Example

```
achievementManager.FetchUserAchievements((res) =>
{
  if (res == Discord.Result.Ok)
  {
    // Count()
    // for() loop
  }
});
```

## CountUserAchievements

Counts the list of a user's achievements for iteration.

Returns ```Int32```.

## Parameters

None

## Example

```
achievementManager.FetchUserAchievements((res) =>
{
  if (res == Discord.Result.Ok)
  {
    Console.WriteLine("User has {0} achievements for this game", achievementManager.CountUserAchievements());
  }
});
```

## GetUserAchievementAt

Gets the user's achievement at a given index of their list of achievements.

Returns ```Discord.UserAchievement```

## Parameters

| name | type | description |
| --- | --- | --- |
| index | Int32 | the index at which to get the achievement |

## Example

```
achievementManager.FetchUserAchievements((res) =>
{
  if (res == Discord.Result.Ok)
  {
    for (int i = 0; i < achievementManager.CountUserAchievements(); i++)
    {
      var achievement = achievementManager.GetUserAchievementAt(i);
      Console.WriteLine("Achievement progress for {0} for user {1}: {2}",
                        achievement.AchievementId,
                        achievement.UserId,
                        achievement.PercentComplete);
    }
  }
});
```

## GetUserAchievement

Gets the user achievement for the given achievement id. If you keep a hardcoded mapping of ```achievement <--> ID``` in your codebase, this will be better than iterating over each achievement. Make sure to call ```FetchUserAchievements()``` first still!

## Parameters

| name | type | description |
| --- | --- | --- |
| achievementId | Int64 | the ID of the achievement to get |

## Example

```
achievementManager.FetchUserAchievements((res) =>
{
  if (res == Discord.Result.Ok)
  {
    var achievement = achievementManager.GetUserAchievement(580159119969878046);
    Console.WriteLine("Achievement progress for {0} for user {1}: {2}",
                      achievement.AchievementId,
                      achievement.UserId,
                      achievement.PercentComplete);
  }
});
```

## OnUserAchievementUpdate

Fires when an achievement is updated for the currently connected user

## Parameters

| name | type | description |
| --- | --- | --- |
| achievement | ref UserAchievement | the achievement that was updated |

## The API Way

Below are the API endpoints and the parameters they accept. If you choose to interface directly with the Discord API, you will need a bot token. This is a special authorization token with which your application can access Discord's HTTP API. Head on over to [your app's settings](https://discord.comhttps://ptb.discord.com/developers/applications), and navigate to the Bot page on the sidebar. From there, mutter abra kadabra and reveal the token. This token is used as an authorization header against our API like so:

```
curl -x POST -h "Authorization: Bot <your token>" https://discord.com/api/some-route/that-does-a-thing
```

Make sure to prepend your token with "Bot"!

Get Achievements[

](#get-achievements)
---------------------------------------

GET/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements

Returns all achievements for the given application. This endpoint has a rate limit of 5 requests per 5 seconds per application.

## Get Achievements

GET

/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements

## Return Object

```
[
  {
    "application_id": "461618159171141643",
    "name": {
      "default": "Win the Game"
    },
    "description": {
      "default": "You won!"
    },
    "secret": false,
    "icon_hash": "52c1636444f64ad7cb5368b158847def",
    "id": "580159119969878046",
    "secure": false
  }
]
```

Get Achievement[

](#get-achievement)
-------------------------------------

GET/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

Returns the given achievement for the given application. This endpoint has a rate limit of 5 requests per 5 seconds per application.

## Get Achievement

GET

/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

## Return Object

```
{
  "application_id": "461618159171141643",
  "name": {
    "default": "Win the Game"
  },
  "description": {
    "default": "You won!"
  },
  "secret": false,
  "icon_hash": "52c1636444f64ad7cb5368b158847def",
  "id": "580159119969878046",
  "secure": false
}
```

Create Achievement[

](#create-achievement)
-------------------------------------------

POST/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements

Creates a new achievement for your application. Applications can have a maximum of 1000 achievements. This endpoint has a rate limit of 5 requests per 5 seconds per application.

## Create Achievement

POST

/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements

## Parameters

| name | type | description |
| --- | --- | --- |
| name | string | the name of the achievement |
| description | string | the user-facing achievement description |
| secret | bool | if the achievement is secret |
| secure | bool | if the achievement is secure |
| icon | ImageType | the icon for the achievement |

## Example: Creating an Achievement

```
{
  "name": {
    "default": "Find the Secret"
  },
  "description": {
    "default": "You found it!"
  },
  "secret": true,
  "secure": false,
  "icon": "data:image/png;base64,base64_data_here"
}
```

## Return Object

```
{
  "application_id": "461618159171141643",
  "name": {
    "default": "Find the Secret"
  },
  "description": {
    "default": "You found it!"
  },
  "secret": true,
  "icon_hash": "52c1636444f64ad7cb5368b158847def",
  "id": "597763781871861018",
  "secure": false
}
```

Update Achievement[

](#update-achievement)
-------------------------------------------

PATCH/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

Updates the achievement for \_\_ALL USERS\_\_. This is NOT to update a single user's achievement progress; this is to edit the UserAchievement itself. This endpoint has a rate limit of 5 requests per 5 seconds per application.

## Update Achievement

PATCH

/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

## Parameters

| name | type | description |
| --- | --- | --- |
| name | string | the name of the achievement |
| description | string | the user-facing achievement description |
| secret | bool | if the achievement is secret |
| secure | bool | if the achievement is secure |
| icon | ImageType | the icon for the achievement |

## Example: Updating an Achievement

```
{
  "name": {
    "default": "How do methods break up?"
  },
  "description": {
    "default": "They stop calling each other!"
  },
  "secret": false,
  "secure": false,
  "icon": "data:image/png;base64,base64_data_here"
}
```

## Return Object

```
{
  "application_id": "461618159171141643",
  "name": {
    "default": "How do methods break up?"
  },
  "description": {
    "default": "They stop calling each other!"
  },
  "secret": false,
  "icon_hash": "7d698b594c691e3d28c92e226b28293c",
  "id": "597638720379682816",
  "secure": false
}
```

Delete Achievement[

](#delete-achievement)
-------------------------------------------

DELETE/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

Deletes the given achievement from your application. This endpoint has a rate limit of 5 requests per 5 seconds per application.

## Delete Achievement

DELETE

/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

## Return Object

```
// 204 No Content
```

Update User Achievement[

](#update-user-achievement)
-----------------------------------------------------

PUT/users/[{user.id}](https://ptb.discord.com/developers/docs/resources/user#user-object)/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

Updates the UserAchievement record for a given user. Use this endpoint to update ```secure``` achievement progress for users. This endpoint has a rate limit of 5 requests per 5 seconds per application.

## Update User Achievement

PUT

/users/[{user.id}](https://ptb.discord.com/developers/docs/resources/user#user-object)/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements/[{achievement.id}](https://ptb.discord.com/developers/docs/game-sdk/achievements#data-models-achievement-struct)

## Parameters

| name | type | description |
| --- | --- | --- |
| percent\_complete | int | the user's progress towards completing the achievement |

## Return Object

```
{}
```

Get User Achievements[

](#get-user-achievements)
-------------------------------------------------

GET/users/@me/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements

Returns a list of achievements for the user whose token you're making the request with. This endpoint will NOT accept the Bearer token for your application generated via the [Client Credentials Grant](https://ptb.discord.com/developers/docs/topics/oauth2#client-credentials-grant). You will need the user's bearer token, gotten via either the [Authorization Code OAuth2 Grant](https://ptb.discord.com/developers/docs/topics/oauth2#authorization-code-grant) or via the SDK with [GetOAuth2Token](https://ptb.discord.com/developers/docs/game-sdk/applications#getoauth2token). This endpoint has a rate limit of 2 requests per 5 seconds per application per user.

This endpoint will not return any achievements marked as ```secret``` that the user has not yet completed.

## Get User Achievements

GET

/users/@me/applications/[{application.id}](https://ptb.discord.com/developers/docs/game-sdk/sdk-starter-guide#get-set-up)/achievements

## Return Object

```
[
  {
    "application_id": "461618159171141643",
    "name": {
      "default": "Win the Game"
    },
    "description": {
      "default": "You won!"
    },
    "secret": false,
    "icon_hash": "52c1636444f64ad7cb5368b158847def",
    "id": "580159119969878046",
    "secure": false
  }
]
```

