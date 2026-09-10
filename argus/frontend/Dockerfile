FROM node:20-alpine

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

# Vite proxies /api to the backend service (compose sets VITE_API_URL=http://backend:8000)
CMD ["npm", "run", "dev"]
