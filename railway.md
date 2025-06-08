# Railway Deployment

Railway is perfect for quick deployments. Here's how:

## 1. Prepare your code
```bash
# Create railway.json
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "python3 api_server.py"
  }
}
```

## 2. Deploy steps
1. Push code to GitHub
2. Connect Railway to your GitHub repo
3. Add environment variables:
   - `OPENAI_API_KEY`
   - `SUPABASE_URL` 
   - `SUPABASE_KEY`
4. Deploy automatically

## 3. Frontend deployment
Deploy the `build/` folder to:
- Vercel
- Netlify  
- Railway static hosting

Update API endpoints to point to your Railway backend URL.