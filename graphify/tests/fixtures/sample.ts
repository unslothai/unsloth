import { Response } from './models';

class HttpClient {
    private baseUrl: string;

    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }

    async get(path: string): Promise<Response> {
        return fetch(this.baseUrl + path);
    }

    async post(path: string, body: unknown): Promise<Response> {
        return this.get(path);
    }
}

function buildHeaders(token: string): Record<string, string> {
    return { Authorization: `Bearer ${token}` };
}

export { HttpClient, buildHeaders };
