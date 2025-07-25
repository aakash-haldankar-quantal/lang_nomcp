# Backend Development Guide:

After cloning the repo.

1.create .env file inside parent directory and copy-paste below.

```plaintext
OPEN_API_KEY="sz1pcA" #Add yours
NOTION_MCP_TOKEN="ntn_3800682831324WTiVCFB2ur9Dp"  #Add yours
NOTION_VERSION="2022-06-29" #I changed
JWT_SECRET="IM ARYAN" #You can keep this as it is or give any string
```


2.Via terminal go to root directory i.e lang_nomcp and run below.
```plaintext
uvicorn apis:app
```
