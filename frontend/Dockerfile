# --- Frontend Dockerfile (multi-stage) ----------------------------------
# Stage 1 "build": use Node to run `npm run build`, which turns the React
# source into plain static HTML/JS/CSS files (no React/Node needed to
# actually serve them).
FROM node:20-slim AS build
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
RUN npm run build

# Stage 2 "serve": a tiny nginx image just serves the static files produced
# above. We don't ship Node or node_modules in the final image at all -
# multi-stage builds keep the deployed image small and free of build-time
# tooling it doesn't need at runtime.
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
