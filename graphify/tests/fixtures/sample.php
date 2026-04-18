<?php

namespace App\Http;

use App\Auth\Authenticator;
use App\Cache\CacheManager;

class ApiClient
{
    private string $baseUrl;
    private Authenticator $auth;

    public function __construct(string $baseUrl)
    {
        $this->baseUrl = $baseUrl;
        $this->auth = new Authenticator();
    }

    public function get(string $path): string
    {
        return $this->fetch($path, 'GET');
    }

    public function post(string $path, string $body): string
    {
        return $this->fetch($path, 'POST');
    }

    private function fetch(string $path, string $method): string
    {
        $token = $this->auth->getToken();
        return $method . ' ' . $this->baseUrl . $path;
    }
}

function parseResponse(string $raw): array
{
    return json_decode($raw, true);
}
