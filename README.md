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

3.Follow the similar link below

<img width="935" height="282" alt="image" src="https://github.com/user-attachments/assets/d9b5c6e2-acaf-4d02-9ffe-8fb5bb29d539" />

write /docs

<img width="1339" height="290" alt="image" src="https://github.com/user-attachments/assets/ce4fb8ba-d884-4477-9134-db1e9a47efe6" />

4.Swagger ui will appear.

<img width="1365" height="622" alt="image" src="https://github.com/user-attachments/assets/6ff60727-ec9f-4246-b811-614d4ee1e9da" />

5.Click on authorise button,enter username & password that's it and click authorised.

6.Start chatting on chat endpoint

<img width="1317" height="584" alt="image" src="https://github.com/user-attachments/assets/b3ec877d-9286-44d0-823b-b345f0bf71ff" />

<img width="1282" height="186" alt="image" src="https://github.com/user-attachments/assets/922a9bbb-57d1-4c92-b8a1-f5aa53110ccc" />



### Additinal Note
1.At any point of time if you get "Token invalid" or "Token expired" then do step 5 followed by 6
