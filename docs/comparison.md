# Comparison

## Similar tools

- [Immich](https://immich.app/) and [PhotoPrism](https://www.photoprism.app/) are self-hosted photo libraries. They lead with faces, albums, maps, mobile apps, and smart search.
- [semantic-photo-search](https://github.com/greisikertuka/semantic-photo-search) shows that CLIP search becomes more useful when filters such as camera, aperture, and ISO can narrow results.

## What find-local does differently

- One local index across text files, PDFs, and photos, with hybrid keyword and embedding ranking.
- A CLI, a local web UI, and an MCP server over the same SQLite index.
- No server deployment, account, or background container.
- A polling watcher keeps the index fresh without a daemon.
- The index and the files stay on the machine.

## When another tool is a better fit

- Use Immich or PhotoPrism for a phone-photo library with faces, albums, and sharing.
- Use find-local for a developer or researcher who wants to search a mixed folder tree from a terminal or an agent.
