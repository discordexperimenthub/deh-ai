# Discord Developer Portal — Documentation — Receiving and Responding

## Interactions

An [Interaction](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object) is the message that your application receives when a user uses an application command or a message component.

For [Slash Commands](https://ptb.discord.com/developers/docs/interactions/application-commands#slash-commands), it includes the values that the user submitted.

For [User Commands](https://ptb.discord.com/developers/docs/interactions/application-commands#user-commands) and [Message Commands](https://ptb.discord.com/developers/docs/interactions/application-commands#message-commands), it includes the resolved user or message on which the action was taken.

For [Message Components](https://ptb.discord.com/developers/docs/interactions/message-components) it includes identifying information about the component that was used. It will also include some metadata about how the interaction was triggered: the ```guild_id```, ```channel```, ```member``` and other fields. You can find all the values in our data models below.

## Interaction Object

## Interaction Structure

| Field | Type | Description |
| --- | --- | --- |
| id | snowflake | ID of the interaction |
| application\_id | snowflake | ID of the application this interaction is for |
| type | [interaction type](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-interaction-type) | Type of interaction |
| data?\* | [interaction data](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-interaction-data) | Interaction data payload |
| guild\_id? | snowflake | Guild that the interaction was sent from |
| channel? | [partial channel](https://ptb.discord.com/developers/docs/resources/channel#channel-object) object | Channel that the interaction was sent from |
| channel\_id? | snowflake | Channel that the interaction was sent from |
| member?\*\* | [guild member](https://ptb.discord.com/developers/docs/resources/guild#guild-member-object) object | Guild member data for the invoking user, including permissions |
| user? | [user](https://ptb.discord.com/developers/docs/resources/user#user-object) object | User object for the invoking user, if invoked in a DM |
| token | string | Continuation token for responding to the interaction |
| version | integer | Read-only property, always ```1``` |
| message? | [message](https://ptb.discord.com/developers/docs/resources/channel#message-object) object | For components, the message they were attached to |
| app\_permissions? | string | Bitwise set of permissions the app or bot has within the channel the interaction was sent from |
| locale?\*\*\* | string | Selected [language](https://ptb.discord.com/developers/docs/reference#locales) of the invoking user |
| guild\_locale? | string | [Guild's preferred locale](https://ptb.discord.com/developers/docs/resources/guild#guild-object), if invoked in a guild |

\* This is always present on application command, message component, and modal submit interaction types. It is optional for future-proofing against new interaction types

\*\* ```member``` is sent when the interaction is invoked in a guild, and ```user``` is sent when invoked in a DM

\*\*\* This is available on all interaction types except PING

## Interaction Type

| Name | Value |
| --- | --- |
| PING | 1 |
| APPLICATION\_COMMAND | 2 |
| MESSAGE\_COMPONENT | 3 |
| APPLICATION\_COMMAND\_AUTOCOMPLETE | 4 |
| MODAL\_SUBMIT | 5 |

## Interaction Data

While the ```data``` field is guaranteed to be present for all [interaction types](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-interaction-type) besides ```PING```, its structure will vary. The following tables detail the inner ```data``` payload for each interaction type.

## Application Command Data Structure

Sent in ```APPLICATION_COMMAND``` and ```APPLICATION_COMMAND_AUTOCOMPLETE``` interactions.

| Field | Type | Description |
| --- | --- | --- |
| id | snowflake | the [```ID```](https://ptb.discord.com/developers/docs/interactions/application-commands#application-command-object-application-command-structure) of the invoked command |
| name | string | the [```name```](https://ptb.discord.com/developers/docs/interactions/application-commands#application-command-object-application-command-structure) of the invoked command |
| type | integer | the [```type```](https://ptb.discord.com/developers/docs/interactions/application-commands#application-command-object-application-command-structure) of the invoked command |
| resolved? | [resolved data](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-resolved-data-structure) | converted users + roles + channels + attachments |
| options?\* | array of [application command interaction data option](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-application-command-interaction-data-option-structure) | the params + values from the user |
| guild\_id? | snowflake | the id of the guild the command is registered to |
| target\_id? | snowflake | id of the user or message targeted by a [user](https://ptb.discord.com/developers/docs/interactions/application-commands#user-commands) or [message](https://ptb.discord.com/developers/docs/interactions/application-commands#message-commands) command |

\* This [can be partial](https://ptb.discord.com/developers/docs/interactions/application-commands#autocomplete) when in response to ```APPLICATION_COMMAND_AUTOCOMPLETE```

## Message Component Data Structure

| Field | Type | Description |
| --- | --- | --- |
| custom\_id | string | the [```custom_id```](https://ptb.discord.com/developers/docs/interactions/message-components#custom-id) of the component |
| component\_type | integer | the [type](https://ptb.discord.com/developers/docs/interactions/message-components#component-object-component-types) of the component |
| values?\* | array of [select option values](https://ptb.discord.com/developers/docs/interactions/message-components#select-menu-object-select-option-structure) | values the user selected in a [select menu](https://ptb.discord.com/developers/docs/interactions/message-components#select-menu-object) component |

\* This is always present for select menu components

## Modal Submit Data Structure

| Field | Type | Description |
| --- | --- | --- |
| custom\_id | string | the [```custom_id```](https://ptb.discord.com/developers/docs/interactions/message-components#custom-id) of the modal |
| components | array of [message components](https://ptb.discord.com/developers/docs/interactions/message-components#message-components) | the values submitted by the user |

## Resolved Data Structure

If data for a Member is included, data for its corresponding User will also be included.

| Field | Type | Description |
| --- | --- | --- |
| users? | Map of Snowflakes to [user](https://ptb.discord.com/developers/docs/resources/user#user-object) objects | the ids and User objects |
| members?\* | Map of Snowflakes to [partial member](https://ptb.discord.com/developers/docs/resources/guild#guild-member-object) objects | the ids and partial Member objects |
| roles? | Map of Snowflakes to [role](https://ptb.discord.com/developers/docs/topics/permissions#role-object) objects | the ids and Role objects |
| channels?\*\* | Map of Snowflakes to [partial channel](https://ptb.discord.com/developers/docs/resources/channel#channel-object) objects | the ids and partial Channel objects |
| messages? | Map of Snowflakes to [partial messages](https://ptb.discord.com/developers/docs/resources/channel#message-object) objects | the ids and partial Message objects |
| attachments? | Map of Snowflakes to [attachment](https://ptb.discord.com/developers/docs/resources/channel#attachment-object) objects | the ids and attachment objects |

\* Partial ```Member``` objects are missing ```user```, ```deaf``` and ```mute``` fields

\*\* Partial ```Channel``` objects only have ```id```, ```name```, ```type``` and ```permissions``` fields. Threads will also have ```thread_metadata``` and ```parent_id``` fields.

## Application Command Interaction Data Option Structure

All options have names, and an option can either be a parameter and input value--in which case ```value``` will be set--or it can denote a subcommand or group--in which case it will contain a top-level key and another array of ```options```.

```value``` and ```options``` are mutually exclusive.

| Field | Type | Description |
| --- | --- | --- |
| name | string | Name of the parameter |
| type | integer | Value of [application command option type](https://ptb.discord.com/developers/docs/interactions/application-commands#application-command-object-application-command-option-type) |
| value? | string, integer, double, or boolean | Value of the option resulting from user input |
| options? | array of [application command interaction data option](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-application-command-interaction-data-option-structure) | Present if this option is a group or subcommand |
| focused? | boolean | ```true``` if this option is the currently focused option for autocomplete |

## Message Interaction Object

This is sent on the [message object](https://ptb.discord.com/developers/docs/resources/channel#message-object) when the message is a response to an Interaction without an existing message.

This means responses to [Message Components](https://ptb.discord.com/developers/docs/interactions/message-components) do not include this property, instead including a [message reference](https://ptb.discord.com/developers/docs/resources/channel#message-reference-object-message-reference-structure) object as components always exist on preexisting messages.

## Message Interaction Structure

| Field | Type | Description |
| --- | --- | --- |
| id | snowflake | ID of the interaction |
| type | [interaction type](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object-interaction-type) | Type of interaction |
| name | string | Name of the [application command](https://ptb.discord.com/developers/docs/interactions/application-commands#application-command-object-application-command-structure), including subcommands and subcommand groups |
| user | [user object](https://ptb.discord.com/developers/docs/resources/user#user-object) | User who invoked the interaction |
| member? | [partial member](https://ptb.discord.com/developers/docs/resources/guild#guild-member-object) object | Member who invoked the interaction in the guild |

## Interactions and Bot Users

We're all used to the way that Discord bots have worked for a long time. You make an application in the developer portal with a bot user, then use that bot's token to connect to the Gateway and make requests to Discord's API.

Interactions bring something entirely new to the table: the ability to interact with an application without needing a bot user in the guild. As you read through this documentation, you'll see that bot tokens are only referenced as a helpful alternative to doing a client credentials auth flow. Responding to interactions does not require a bot token.

In many cases, you may still need a bot user. If you need to receive gateway events, or need to interact with other parts of our API (like fetching a guild, or a channel, or updating permissions on a user), those actions are all still tied to having a bot token. However, if you don't need any of those things, you never have to add a bot user to your application at all.

Welcome to the new world.

## Receiving an Interaction

When a user interacts with your app, your app will receive an [Interaction](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object). Your app can receive an interaction in one of two ways:

*   Via [Interaction Create](https://ptb.discord.com/developers/docs/topics/gateway-events#interaction-create) gateway event
*   Via outgoing webhook

These two methods are mutually exclusive; you can only receive Interactions one of the two ways. The ```INTERACTION_CREATE``` [Gateway Event](https://ptb.discord.com/developers/docs/topics/gateway-events#interaction-create) may be handled by connected clients, while the webhook method detailed below does not require a connected client.

In your application in the Developer Portal, there is a field on the main page called "Interactions Endpoint URL". If you want to receive Interactions via outgoing webhook, you can set your URL in this field. In order for the URL to be valid, you must be prepared for two things ahead of time:

These steps are only necessary for webhook-based Interactions. It is not required for receiving them over the gateway.

1.  Your endpoint must be prepared to ACK a ```PING``` message
2.  Your endpoint must be set up to properly handle signature headers--more on that in [Security and Authorization](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#security-and-authorization)

If either of these are not complete, we will not validate your URL and it will fail to save.

When you attempt to save a URL, we will send a ```POST``` request to that URL with a ```PING``` payload. The ```PING``` payload has a ```type: 1```. So, to properly ACK the payload, return a ```200``` response with a payload of ```type: 1```:

```
@app.route('/', methods=['POST'])
def my_command():
    if request.json["type"] == 1:
        return jsonify({
            "type": 1
        })
```

You'll also need to properly set up [Security and Authorization](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#security-and-authorization) on your endpoint for the URL to be accepted. Once both of those are complete and your URL has been saved, you can start receiving Interactions via webhook! At this point, your app will no longer receive Interactions over the gateway. If you want to receive them over the gateway again, simply delete your URL.

An [Interaction](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object) includes metadata to aid your application in handling it as well as ```data``` specific to the interaction type. You can find samples for each interaction type on their respective pages:

*   [Slash Commands](https://ptb.discord.com/developers/docs/interactions/application-commands#slash-commands-example-interaction)
*   [User Commands](https://ptb.discord.com/developers/docs/interactions/application-commands#user-commands-example-interaction)
*   [Message Commands](https://ptb.discord.com/developers/docs/interactions/application-commands#message-commands-example-interaction)
*   [Message Components](https://ptb.discord.com/developers/docs/interactions/message-components#component-interaction-object-sample-component-interaction)
*   [Select Menu Message Components](https://ptb.discord.com/developers/docs/interactions/message-components#select-menu-object-select-menu-interaction)

An explanation of all the fields can be found in our [data models](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object).

Now that you've gotten the data from the user, it's time to respond to them.

## Responding to an Interaction

Interactions--both receiving and responding--are webhooks under the hood. So responding to an Interaction is just like sending a webhook request!

There are a number of ways you can respond to an interaction:

## Interaction Response Object

## Interaction Response Structure

| Field | Type | Description |
| --- | --- | --- |
| type | [interaction callback type](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-response-object-interaction-callback-type) | the type of response |
| data? | [interaction callback data](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-response-object-interaction-callback-data-structure) | an optional response message |

## Interaction Callback Type

| Name | Value | Description |
| --- | --- | --- |
| PONG | 1 | ACK a ```Ping``` |
| CHANNEL\_MESSAGE\_WITH\_SOURCE | 4 | respond to an interaction with a message |
| DEFERRED\_CHANNEL\_MESSAGE\_WITH\_SOURCE | 5 | ACK an interaction and edit a response later, the user sees a loading state |
| DEFERRED\_UPDATE\_MESSAGE\* | 6 | for components, ACK an interaction and edit the original message later; the user does not see a loading state |
| UPDATE\_MESSAGE\* | 7 | for components, edit the message the component was attached to |
| APPLICATION\_COMMAND\_AUTOCOMPLETE\_RESULT | 8 | respond to an autocomplete interaction with suggested choices |
| MODAL\*\* | 9 | respond to an interaction with a popup modal |

\* Only valid for [component-based](https://ptb.discord.com/developers/docs/interactions/message-components) interactions

\*\* Not available for MODAL\_SUBMIT and PING interactions.

## Interaction Callback Data Structure

## Messages

Not all message fields are currently supported.

| Field | Type | Description |
| --- | --- | --- |
| tts? | boolean | is the response TTS |
| content? | string | message content |
| embeds? | array of [embeds](https://ptb.discord.com/developers/docs/resources/channel#embed-object) | supports up to 10 embeds |
| allowed\_mentions? | [allowed mentions](https://ptb.discord.com/developers/docs/resources/channel#allowed-mentions-object) | [allowed mentions](https://ptb.discord.com/developers/docs/resources/channel#allowed-mentions-object) object |
| flags? | integer | [message flags](https://ptb.discord.com/developers/docs/resources/channel#message-object-message-flags) combined as a [bitfield](https://en.wikipedia.org/wiki/Bit_field) (only ```SUPPRESS_EMBEDS``` and ```EPHEMERAL``` can be set) |
| components? | array of [components](https://ptb.discord.com/developers/docs/interactions/message-components) | message components |
| attachments? \* | array of partial [attachment](https://ptb.discord.com/developers/docs/resources/channel#attachment-object) objects | attachment objects with filename and description |

\* See [Uploading Files](https://ptb.discord.com/developers/docs/reference#uploading-files) for details.

## Autocomplete

| Field | Type | Description |
| --- | --- | --- |
| choices | array of [choices](https://ptb.discord.com/developers/docs/interactions/application-commands#application-command-object-application-command-option-choice-structure) | autocomplete choices (max of 25 choices) |

## Modal

Support for components in modals is currently limited to type 4 (Text Input).

| Field | Type | Description |
| --- | --- | --- |
| custom\_id | string | a developer-defined identifier for the modal, max 100 characters |
| title | string | the title of the popup modal, max 45 characters |
| components | array of [components](https://ptb.discord.com/developers/docs/interactions/message-components) | between 1 and 5 (inclusive) components that make up the modal |

While interaction responses and followups are webhooks, they respect @everyone's ability to ping @everyone / @here . Nonetheless if your application responds with user data, you should still use [```allowed_mentions```](https://ptb.discord.com/developers/docs/resources/channel#allowed-mentions-object) to filter which mentions in the content actually ping. Other differences include the ability to send named links in the message content (```[text](url)```).

When responding to an interaction received via webhook, your server can simply respond to the received ```POST``` request. You'll want to respond with a ```200``` status code (if everything went well), as well as specifying a ```type``` and ```data```, which is an [Interaction Response](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-response-object) object:

```
@app.route('/', methods=['POST'])
def my_command():
    if request.json["type"] == 1:
        return jsonify({
            "type": 1
        })

    else:
        return jsonify({
            "type": 4,
            "data": {
                "tts": False,
                "content": "Congrats on sending your command!",
                "embeds": [],
                "allowed_mentions": { "parse": [] }
            }
        })
```

If you are receiving Interactions over the gateway, you will also need to respond via HTTP. Responses to Interactions are not sent as commands over the gateway.

To respond to a gateway Interaction, make a ```POST``` request like this. ```interaction_id``` is the unique id of that individual Interaction from the received payload. ```interaction_token``` is the unique token for that interaction from the received payload. This endpoint is only valid for Interactions received over the gateway. Otherwise, respond to the ```POST``` request to issue an initial response.

```
import requests

url = "https://discord.com/api/v10/interactions/<interaction_id>/<interaction_token>/callback"

json = {
    "type": 4,
    "data": {
        "content": "Congrats on sending your command!"
    }
}
r = requests.post(url, json=json)
```

Interaction ```tokens``` are valid for 15 minutes and can be used to send followup messages but you must send an initial response within 3 seconds of receiving the event. If the 3 second deadline is exceeded, the token will be invalidated.

## Followup Messages

Sometimes, your bot will want to send followup messages to a user after responding to an interaction. Or, you may want to edit your original response. Whether you receive Interactions over the gateway or by outgoing webhook, you can use the following endpoints to edit your initial response or send followup messages:

*   [```PATCH /webhooks/<application_id>/<interaction_token>/messages/@original```](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#edit-original-interaction-response) to edit your initial response to an Interaction
*   [```DELETE /webhooks/<application_id>/<interaction_token>/messages/@original```](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#delete-original-interaction-response) to delete your initial response to an Interaction
*   [```POST /webhooks/<application_id>/<interaction_token>```](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#create-followup-message) to send a new followup message
*   [```PATCH /webhooks/<application_id>/<interaction_token>/messages/<message_id>```](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#edit-followup-message) to edit a message sent with that ```token```

Interactions webhooks share the same rate limit properties as normal webhooks.

Interaction tokens are valid for 15 minutes, meaning you can respond to an interaction within that amount of time.

## Security and Authorization

Check out our [Community Resources](https://ptb.discord.com/developers/docs/topics/community-resources#interactions) for libraries to help you with security in your language of choice.

The internet is a scary place, especially for people hosting open, unauthenticated endpoints. If you are receiving Interactions via outgoing webhook, there are some security steps you must take before your app is eligible to receive requests.

Every Interaction is sent with the following headers:

*   ```X-Signature-Ed25519``` as a signature
*   ```X-Signature-Timestamp``` as a timestamp

Using your favorite security library, you must validate the request each time you receive an [interaction](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object). If the signature fails validation, respond with a ```401``` error code. Here's a couple code examples:

```
const nacl = require('tweetnacl');

// Your public key can be found on your application in the Developer Portal
const PUBLIC_KEY = 'APPLICATION_PUBLIC_KEY';

const signature = req.get('X-Signature-Ed25519');
const timestamp = req.get('X-Signature-Timestamp');
const body = req.rawBody; // rawBody is expected to be a string, not raw bytes

const isVerified = nacl.sign.detached.verify(
  Buffer.from(timestamp + body),
  Buffer.from(signature, 'hex'),
  Buffer.from(PUBLIC_KEY, 'hex')
);

if (!isVerified) {
  return res.status(401).end('invalid request signature');
}
```

```
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

# Your public key can be found on your application in the Developer Portal
PUBLIC_KEY = 'APPLICATION_PUBLIC_KEY'

verify_key = VerifyKey(bytes.fromhex(PUBLIC_KEY))

signature = request.headers["X-Signature-Ed25519"]
timestamp = request.headers["X-Signature-Timestamp"]
body = request.data.decode("utf-8")

try:
    verify_key.verify(f'{timestamp}{body}'.encode(), bytes.fromhex(signature))
except BadSignatureError:
    abort(401, 'invalid request signature')
```

If you are not properly validating this signature header, we will not allow you to save your interactions URL in the Dev Portal. We will also do automated, routine security checks against your endpoint, including purposefully sending you invalid signatures. If you fail the validation, we will remove your interactions URL in the future and alert you via email and System DM.

We highly recommend checking out our [Community Resources](https://ptb.discord.com/developers/docs/topics/community-resources#interactions) and the libraries found there. They not only provide typing for Interactions data models, but also include decorators for API frameworks like Flask and Express to make validation easy.

## Endpoints

The endpoints below are not bound to the application's [Global Rate Limit](https://ptb.discord.com/developers/docs/topics/rate-limits#global-rate-limit).

Create Interaction Response[

](#create-interaction-response)
-------------------------------------------------------------

POST/interactions/[{interaction.id}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/callback

Create a response to an Interaction from the gateway. Body is an [interaction response](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-response-object). Returns ```204 No Content```.

This endpoint also supports file attachments similar to the webhook endpoints. Refer to [Uploading Files](https://ptb.discord.com/developers/docs/reference#uploading-files) for details on uploading files and ```multipart/form-data``` requests.

Get Original Interaction Response[

](#get-original-interaction-response)
-------------------------------------------------------------------------

GET/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/@original

Returns the initial Interaction response. Functions the same as [Get Webhook Message](https://ptb.discord.com/developers/docs/resources/webhook#get-webhook-message).

Edit Original Interaction Response[

](#edit-original-interaction-response)
---------------------------------------------------------------------------

PATCH/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/@original

Edits the initial Interaction response. Functions the same as [Edit Webhook Message](https://ptb.discord.com/developers/docs/resources/webhook#edit-webhook-message).

Delete Original Interaction Response[

](#delete-original-interaction-response)
-------------------------------------------------------------------------------

DELETE/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/@original

Deletes the initial Interaction response. Returns ```204 No Content``` on success.

Create Followup Message[

](#create-followup-message)
-----------------------------------------------------

POST/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)

Create a followup message for an Interaction. Functions the same as [Execute Webhook](https://ptb.discord.com/developers/docs/resources/webhook#execute-webhook), but ```wait``` is always true. The ```thread_id```, ```avatar_url```, and ```username``` parameters are not supported when using this endpoint for interaction followups.

```flags``` can be set to ```64``` to mark the message as ephemeral, except when it is the first followup message to a deferred Interactions Response. In that case, the ```flags``` field will be ignored, and the ephemerality of the message will be determined by the ```flags``` value in your original ACK.

Get Followup Message[

](#get-followup-message)
-----------------------------------------------

GET/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/[{message.id}](https://ptb.discord.com/developers/docs/resources/channel#message-object)

Returns a followup message for an Interaction. Functions the same as [Get Webhook Message](https://ptb.discord.com/developers/docs/resources/webhook#get-webhook-message).

Edit Followup Message[

](#edit-followup-message)
-------------------------------------------------

PATCH/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/[{message.id}](https://ptb.discord.com/developers/docs/resources/channel#message-object)

Edits a followup message for an Interaction. Functions the same as [Edit Webhook Message](https://ptb.discord.com/developers/docs/resources/webhook#edit-webhook-message).

Delete Followup Message[

](#delete-followup-message)
-----------------------------------------------------

DELETE/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/[{message.id}](https://ptb.discord.com/developers/docs/resources/channel#message-object)

Deletes a followup message for an Interaction. Returns ```204 No Content``` on success.

## Create Interaction Response

POST

/interactions/[{interaction.id}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/callback

## Get Original Interaction Response

GET

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/@original

## Edit Original Interaction Response

PATCH

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/@original

## Delete Original Interaction Response

DELETE

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/@original

## Create Followup Message

POST

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)

## Get Followup Message

GET

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/[{message.id}](https://ptb.discord.com/developers/docs/resources/channel#message-object)

## Edit Followup Message

PATCH

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/[{message.id}](https://ptb.discord.com/developers/docs/resources/channel#message-object)

## Delete Followup Message

DELETE

/webhooks/[{application.id}](https://ptb.discord.com/developers/docs/resources/application#application-object)/[{interaction.token}](https://ptb.discord.com/developers/docs/interactions/receiving-and-responding#interaction-object)/messages/[{message.id}](https://ptb.discord.com/developers/docs/resources/channel#message-object)

