0. python version should be 3.10.x~3.12.x  
1. Download, or use git clone  
2. Create a folder for this pipeline with your preferred name such as "GenVal."
3. Open with git bash and go to the folder you created
4. Activate virtual Environment there

```bash
source .venv/Scripts/activate
```

5. Install the Dependencies
```bash
pip install -r requirements.txt
```
6. create a <.env> file, you can rename a txt file or use the command
```
vim .env
```

7. The content should be in a text file
```
OPENAI_API_KEY=your_openai_key
